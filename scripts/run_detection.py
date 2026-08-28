#!/usr/bin/env python
"""Detection grid: attacks x perceptual hashes x detectors x poison rates.

Writes one row per cell to ``results/detection.csv``:
    attack, rate, hash, n_bits, detector, auc, ap, tpr@1fpr, tpr@5fpr,
    auc_target_class, elimination_rate, sacrifice_rate, ...

No model is trained anywhere in this script -- that is the whole point of the
defense being evaluated.
"""

from __future__ import annotations

import argparse
import itertools
import os
import os.path as osp
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))

from phash_defense import attacks as A  # noqa: E402
from phash_defense import detectors as D  # noqa: E402
from phash_defense import hashes as H  # noqa: E402
from phash_defense.metrics import detection_metrics  # noqa: E402

RESULTS = osp.abspath(osp.join(osp.dirname(__file__), "..", "results"))

MAIN_ATTACKS = ["label_flip_only", "badnets", "badnets_1px", "blended", "blended_a05",
                "blended_a02", "wanet", "refool", "batt", "adaptive_patch",
                "clean_label_badnets", "stego", "phash_adaptive", "phash_adaptive_ss"]
QUICK_ATTACKS = ["badnets", "blended", "wanet", "phash_adaptive"]
QUICK_HASHES = ["ahash", "phash", "dhash", "blockhash", "rhash16", "rhash_energy"]
QUICK_DETECTORS = ["bit_llr", "hamming_knn", "block_collision"]


def evaluate_cell(ps, hash_name, detector_names, max_n, rule, k, frac, stability_cache):
    spec = H.HASHES[hash_name]
    t0 = time.time()
    bits = H.compute(hash_name, ps.images)
    t_hash = time.time() - t0
    rows = []
    for det in detector_names:
        dspec = D.DETECTORS[det]
        params = {}
        if dspec.needs_images:
            key = (hash_name, id(ps))
            if key not in stability_cache:
                stability_cache[key] = D.stability_scores(ps.images, hash_name)
            params["stability"] = stability_cache[key]
        if det in ("hamming_knn", "hamming_ball"):
            params["max_n"] = max_n
        t1 = time.time()
        try:
            scores = D.run(det, bits, ps.labels, layout=spec.layout, **params)
        except Exception as exc:  # noqa: BLE001
            print(f"    !! {hash_name}/{det} failed: {exc}")
            continue
        if scores is None:
            continue
        keep = D.threshold_mask(scores, ps.labels, rule=rule, k=k, frac=frac)
        m = detection_metrics(ps.poison_mask, scores, labels=ps.labels,
                              target_class=ps.y_target, keep_mask=keep)
        m.update(dict(attack=ps.name, rate=round(ps.poison_rate, 5), hash=hash_name,
                      n_bits=int(bits.shape[1]), family=spec.family,
                      threshold=spec.threshold, detector=det, rule=rule,
                      t_hash_s=round(t_hash, 3), t_detect_s=round(time.time() - t1, 3)))
        rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks", nargs="*", default=None)
    ap.add_argument("--hashes", nargs="*", default=None)
    ap.add_argument("--detectors", nargs="*", default=None)
    ap.add_argument("--rates", nargs="*", type=float, default=[0.05])
    ap.add_argument("--y-target", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-n", type=int, default=1500,
                    help="reference-set size for the O(n^2) detectors")
    ap.add_argument("--rule", default="mad")
    ap.add_argument("--k", type=float, default=3.5)
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=osp.join(RESULTS, "detection.csv"))
    args = ap.parse_args()

    attacks = args.attacks or (QUICK_ATTACKS if args.quick else MAIN_ATTACKS)
    hash_names = args.hashes or (QUICK_HASHES if args.quick else list(H.HASHES))
    detector_names = args.detectors or (QUICK_DETECTORS if args.quick else list(D.DETECTORS))
    rates = args.rates

    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    for attack, rate in itertools.product(attacks, rates):
        t0 = time.time()
        ps = A.build(attack, poisoned_rate=rate, y_target=args.y_target, seed=args.seed)
        print(f"[{attack} @ {rate}] {len(ps)} samples, poison={ps.poison_mask.sum()} "
              f"({time.time() - t0:.1f}s)", flush=True)
        cache = {}
        for h in hash_names:
            r = evaluate_cell(ps, h, detector_names, args.max_n, args.rule, args.k,
                              args.frac, cache)
            rows.extend(r)
            if r:
                best = max(r, key=lambda x: x["auc"] if np.isfinite(x["auc"]) else -1)
                print(f"    {h:14s} best={best['detector']:16s} auc={best['auc']:.4f} "
                      f"tpr@1={best['tpr@1fpr']:.3f}", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(args.out, index=False)
    print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
