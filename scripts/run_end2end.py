#!/usr/bin/env python
"""End-to-end evaluation of the purification defense.

For every attack: train on the poisoned set, purify it with the perceptual-hash
defense, retrain, and report benign accuracy (BA) and attack success rate (ASR).
A benign reference run is added once.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))

from phash_defense import attacks as A  # noqa: E402
from phash_defense import detectors as D  # noqa: E402
from phash_defense import hashes as H  # noqa: E402
from phash_defense.data import cifar10  # noqa: E402
from phash_defense.metrics import detection_metrics  # noqa: E402
from phash_defense.train import train_model  # noqa: E402

RESULTS = osp.abspath(osp.join(osp.dirname(__file__), "..", "results"))

#: PHash-Sieve: a sample is dropped if *any* stage flags it. Stage 1 is the
#: block-collision sieve with the robust MAD threshold (catches fixed patches and
#: any trigger that fixes a spatial tile), stage 2 is the bit-LLR statistic with
#: the 2-means threshold (catches global and steganographic triggers, where the
#: shift is distributed over many bits and no single tile is fixed).
DEFAULT_CONFIG = {
    "stages": [
        {"hashes": ["rhash32"], "detector": "block_collision", "rule": "mad", "k": 3.5},
        {"hashes": ["rhash32"], "detector": "bit_llr", "rule": "cluster", "k": 3.5},
    ],
    "name": "rhash32/block_collision+bit_llr",
}


def purify(ps, cfg):
    """Run the defense and return (keep_mask, scores, detection metrics)."""
    keep = np.ones(len(ps), dtype=bool)
    first_scores = None
    for stage in cfg["stages"]:
        per_hash = []
        for name in stage["hashes"]:
            spec = H.HASHES[name]
            bits = H.compute(name, ps.images)
            params = {}
            if D.DETECTORS[stage["detector"]].needs_images:
                params["stability"] = D.stability_scores(ps.images, name)
            s = D.run(stage["detector"], bits, ps.labels, layout=spec.layout, **params)
            if s is not None:
                per_hash.append(s)
        if not per_hash:
            continue
        scores = per_hash[0] if len(per_hash) == 1 else D.ensemble(per_hash)
        first_scores = scores if first_scores is None else first_scores
        keep &= D.threshold_mask(scores, ps.labels, rule=stage["rule"],
                                 k=stage.get("k", 3.5))
    m = detection_metrics(ps.poison_mask, first_scores, labels=ps.labels,
                          target_class=ps.y_target, keep_mask=keep)
    return keep, first_scores, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks", nargs="*", default=["badnets", "blended", "wanet"])
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--y-target", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--config", default=None, help="JSON file overriding the defense config")
    ap.add_argument("--skip-benign", action="store_true")
    ap.add_argument("--out", default=osp.join(RESULTS, "end2end.csv"))
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG if args.config is None else json.load(open(args.config))
    os.makedirs(RESULTS, exist_ok=True)
    rows = []

    tr_imgs, tr_labels = cifar10(train=True)
    te_imgs, te_labels = cifar10(train=False)

    if not args.skip_benign:
        print("=== benign reference ===", flush=True)
        res, _ = train_model(tr_imgs, tr_labels, te_imgs, te_labels,
                             epochs=args.epochs, device=args.device, seed=args.seed)
        rows.append(dict(attack="none", setting="benign", **res))
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"    BA={res['ba']:.4f}", flush=True)

    for attack in args.attacks:
        print(f"\n=== {attack} (rate={args.rate}) ===", flush=True)
        ps = A.build(attack, poisoned_rate=args.rate, y_target=args.y_target, seed=args.seed)
        pte = A.build(attack, poisoned_rate=1.0, y_target=args.y_target,
                      seed=args.seed, split="test")

        print("  [1/2] training on the poisoned set", flush=True)
        res, _ = train_model(ps.images, ps.labels, te_imgs, te_labels, poisoned_test=pte,
                             epochs=args.epochs, device=args.device, seed=args.seed)
        rows.append(dict(attack=attack, setting="poisoned", rate=args.rate, **res))
        print(f"    BA={res['ba']:.4f} ASR={res['asr']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

        keep, scores, m = purify(ps, cfg)
        print(f"  purification: removed {int((~keep).sum())} samples, "
              f"elimination={m['elimination_rate']:.3f} sacrifice={m['sacrifice_rate']:.3f} "
              f"auc={m['auc']:.4f}", flush=True)

        print("  [2/2] training on the purified set", flush=True)
        res2, _ = train_model(ps.images[keep], ps.labels[keep], te_imgs, te_labels,
                              poisoned_test=pte, epochs=args.epochs, device=args.device,
                              seed=args.seed)
        rows.append(dict(attack=attack, setting="purified", rate=args.rate,
                         defense=cfg.get("name", "phash-sieve"),
                         elimination_rate=m["elimination_rate"],
                         sacrifice_rate=m["sacrifice_rate"],
                         detection_auc=m["auc"], **res2))
        print(f"    BA={res2['ba']:.4f} ASR={res2['asr']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
