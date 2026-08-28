#!/usr/bin/env python
"""Purification profile of the deployed PHash-Sieve configurations.

For every attack, reports how much poison each sieve variant removes and how
much clean data it sacrifices. This is the operating-point table: the detection
grid measures ranking quality (AUC), this one measures what the defense actually
does when it has to commit to a decision.

Writes results/purification.csv.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))

from phash_defense import attacks as A  # noqa: E402
from phash_defense import detectors as D  # noqa: E402
from phash_defense import hashes as H  # noqa: E402
from phash_defense.metrics import detection_metrics  # noqa: E402

RESULTS = osp.abspath(osp.join(osp.dirname(__file__), "..", "results"))

STAGE_B = {"hash": "rhash32", "detector": "block_collision", "rule": "mad", "k": 3.5}
STAGE_L = {"hash": "rhash32", "detector": "bit_llr", "rule": "cluster", "k": 3.5}
STAGE_S = {"hash": "rhash32", "detector": "hash_stability", "rule": "mad", "k": 3.5}

CONFIGS = {
    "Sieve-B (block_collision, MAD)": [STAGE_B],
    "Sieve-L (bit_llr, 2-means)": [STAGE_L],
    "Sieve-S (hash_stability, MAD)": [STAGE_S],
    "PHash-Sieve (B + L)": [STAGE_B, STAGE_L],
}

ATTACKS = ["label_flip_only", "badnets", "badnets_1px", "blended", "blended_a05",
           "blended_a02", "wanet", "refool", "batt", "adaptive_patch",
           "clean_label_badnets", "stego", "phash_adaptive", "phash_adaptive_ss"]


def stage_scores(ps, stage, cache):
    key = (stage["hash"], stage["detector"])
    if key in cache:
        return cache[key]
    spec = H.HASHES[stage["hash"]]
    bits = H.compute(stage["hash"], ps.images)
    params = {}
    if D.DETECTORS[stage["detector"]].needs_images:
        params["stability"] = D.stability_scores(ps.images, stage["hash"])
    s = D.run(stage["detector"], bits, ps.labels, layout=spec.layout, **params)
    cache[key] = s
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks", nargs="*", default=ATTACKS)
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--y-target", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=osp.join(RESULTS, "purification.csv"))
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    for attack in args.attacks:
        ps = A.build(attack, poisoned_rate=args.rate, y_target=args.y_target, seed=args.seed)
        cache = {}
        for label, stages in CONFIGS.items():
            keep = np.ones(len(ps), dtype=bool)
            first = None
            for st in stages:
                s = stage_scores(ps, st, cache)
                if s is None:
                    continue
                first = s if first is None else first
                keep &= D.threshold_mask(s, ps.labels, rule=st["rule"], k=st["k"])
            m = detection_metrics(ps.poison_mask, first, labels=ps.labels,
                                  target_class=ps.y_target, keep_mask=keep)
            rows.append(dict(attack=attack, config=label,
                             elimination_rate=m["elimination_rate"],
                             sacrifice_rate=m["sacrifice_rate"],
                             residual_poison_rate=m["residual_poison_rate"],
                             n_removed=m["n_removed"],
                             auc_target_class_best=m["auc_target_class_best"]))
            print(f"{attack:22s} {label:32s} elim={m['elimination_rate']:.3f} "
                  f"sacr={m['sacrifice_rate']:.3f} resid={m['residual_poison_rate']:.3f}",
                  flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    print("\n=== elimination rate ===")
    print(df.pivot_table(index="attack", columns="config",
                         values="elimination_rate").round(3).to_string())
    print("\n=== sacrifice rate ===")
    print(df.pivot_table(index="attack", columns="config",
                         values="sacrifice_rate").round(3).to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
