#!/usr/bin/env python
"""MPC suitability benchmark.

1. Correctness gate -- the fixed-point MPC formulation of every hash is compared
   bit-for-bit with the plaintext implementation.
2. Per-image online cost of each hash under 3PC-replicated and 2PC-SPDZ-style
   protocols, on LAN and WAN.
3. Cost of the dataset-level detectors on a 50 000-sample CIFAR-10 training set.
4. Cost of the baselines a classical defense would need (pixel k-NN, k-means /
   activation clustering, ResNet-18 inference, ResNet-18 training).

Outputs: results/mpc_correctness.csv, results/mpc_hash_cost.csv,
         results/mpc_pipeline_cost.csv
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))

from phash_defense import hashes as H  # noqa: E402
from phash_defense.mpc import circuits as C  # noqa: E402
from phash_defense.mpc import costs as K  # noqa: E402
from phash_defense.mpc.protocols import PROTOCOLS, cost_report  # noqa: E402

RESULTS = osp.abspath(osp.join(osp.dirname(__file__), "..", "results"))
N_TRAIN = 50000
N_CLASSES = 10
N_PER_CLASS = N_TRAIN // N_CLASSES


def sample_images(n=512):
    """Real CIFAR-10 images if they are already on disk, random ones otherwise.

    ``download=False`` on purpose: this benchmark must never start a download.
    """
    try:
        import torchvision

        from phash_defense.data import DATA_DIR

        ds = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=False)
        return np.asarray(ds.data[:n], dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        print(f"(CIFAR-10 unavailable: {exc}; falling back to random images)")
        return np.random.default_rng(0).integers(0, 256, (n, 32, 32, 3), dtype=np.uint8)


def resnet18_profile():
    """Multiply-accumulates and ReLU units of one ResNet-18 forward pass (CIFAR)."""
    import torch
    import torch.nn as nn

    sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..", "BackdoorBox")))
    import core

    model = core.models.ResNet(18, 10).eval()
    macs = {"n": 0}
    relus = {"n": 0}

    def conv_hook(mod, inp, out):
        macs["n"] += int(out.numel() * mod.in_channels * mod.kernel_size[0]
                         * mod.kernel_size[1] / mod.groups)

    def lin_hook(mod, inp, out):
        macs["n"] += int(out.numel() * mod.in_features)

    def bn_hook(mod, inp, out):
        relus["n"] += int(out.numel())     # every BN output is followed by a ReLU

    hs = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hs.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hs.append(m.register_forward_hook(lin_hook))
        elif isinstance(m, nn.BatchNorm2d):
            hs.append(m.register_forward_hook(bn_hook))
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32))
    for h in hs:
        h.remove()
    return macs["n"], relus["n"]


def correctness_table(images, frac_bits_list=(8, 12, 16, 20)):
    rows = []
    for name in C.MPC_HASHES:
        for f in frac_bits_list:
            c, agree, exact = C.verify(name, images, frac_bits=f)
            rows.append(dict(hash=name, frac_bits=f, bit_agreement=agree,
                             exact_match=exact, n_images=len(images)))
            print(f"  {name:14s} f={f:2d} agreement={agree:.6f} exact={exact:.4f}", flush=True)
    return pd.DataFrame(rows)


def hash_cost_table(images, frac_bits=16):
    rows = []
    variants = [(n, {}) for n in C.MPC_HASHES]
    variants += [("mhash", {"median": "sort"}), ("phash", {"median": "sort"})]
    for name, kw in variants:
        c, agree, exact = C.verify(name, images, frac_bits=frac_bits, **kw)
        per_image = c.scaled(1)
        n = len(images)
        # normalise the counts to a single image
        for attr in ("n_mult", "n_and", "n_b2a", "n_trunc", "n_local_mac"):
            setattr(per_image, attr, int(getattr(c, attr) / n))
        per_image.n_cmp = {w: int(v / n) for w, v in c.n_cmp.items()}
        per_image.n_eq = {w: int(v / n) for w, v in c.n_eq.items()}
        per_image.rounds = c.rounds
        label = name + ("-" + kw["median"] if kw else "")
        spec = H.HASHES[C.MPC_HASHES[name][1]]
        summary = {f"circ_{k}": v for k, v in per_image.summary().items()}
        for proto in PROTOCOLS:
            for net in ("LAN", "WAN"):
                r = cost_report(per_image, protocol=proto, network=net)
                rows.append(dict(hash=label, n_bits=spec.n_bits, family=spec.family,
                                 threshold=spec.threshold, bit_agreement=agree,
                                 **summary, **r))
        print(f"  {label:18s} bits={spec.n_bits:4d} cmp/img={sum(per_image.n_cmp.values()):6d} "
              f"rounds={per_image.rounds:5d} agree={agree:.4f}", flush=True)
    return pd.DataFrame(rows)


def deployed_configurations(images, frac_bits=16, n=N_TRAIN):
    """Total cost of the three PHash-Sieve variants actually recommended in the
    report: hashing with rhash32 (1024 bits) plus one detector."""
    c, _, _ = C.verify("rhash32", images[:16], frac_bits=frac_bits)
    per = c.scaled(1)
    for attr in ("n_mult", "n_and", "n_b2a", "n_trunc", "n_local_mac"):
        setattr(per, attr, int(getattr(c, attr) / 16))
    per.n_cmp = {w: int(v / 16) for w, v in c.n_cmp.items()}
    hashing = per.scaled(n)
    hashing.rounds = per.rounds          # all images hash in the same rounds

    B, npc = 1024, n // N_CLASSES
    stages = {
        "Sieve-B": [hashing, K.block_collision(npc, 225, 16, N_CLASSES)],
        "Sieve-L": [hashing, K.bit_llr(n, B, N_CLASSES)],
        "Sieve-S": [hashing, K.hash_stability(n, B)],
        "PHash-Sieve (B + L)": [hashing, K.block_collision(npc, 225, 16, N_CLASSES),
                                K.bit_llr(n, B, N_CLASSES)],
    }
    rows = []
    for label, parts in stages.items():
        for proto in PROTOCOLS:
            for net in ("LAN", "WAN"):
                reps = [cost_report(p, protocol=proto, network=net, n_items=n) for p in parts]
                rows.append(dict(component=f"deployed/{label}",
                                 rounds=sum(r["rounds"] for r in reps),
                                 online_MB=sum(r["online_MB"] for r in reps),
                                 time_ms=sum(r["time_ms"] for r in reps),
                                 online_bits=sum(r["online_bits"] for r in reps),
                                 protocol=PROTOCOLS[proto].name, network=net))
        r = [x for x in rows if x["network"] == "LAN" and x["protocol"].startswith("3PC")][-1]
        print(f"  {label:22s} {r['online_MB'] / 1000:8.2f} GB  {r['rounds']:6d} rounds  "
              f"{r['time_ms'] / 1000:8.2f} s (3PC/LAN)", flush=True)
    return pd.DataFrame(rows)


def pipeline_cost_table(macs, relus, hash_bits=256, n=N_TRAIN):
    """Dataset-level cost of every detector and of the classical baselines."""
    entries = {
        "phash-sieve/bit_llr (B=1024)": K.bit_llr(n, 1024, N_CLASSES),
        "phash-sieve/block_collision (225 tiles x 16 bit, B=1024)":
            K.block_collision(N_TRAIN // N_CLASSES, 225, 16, N_CLASSES),
        "phash-sieve/hash_stability (B=1024)": K.hash_stability(n, 1024),
        "phash-sieve/hamming_knn (B=1024, k=10)":
            K.hamming_knn(N_TRAIN // N_CLASSES, 1024, 10, N_CLASSES),
        "phash-sieve/bit_llr (B=256)": K.bit_llr(n, hash_bits, N_CLASSES),
        "phash-sieve/bit_llr private weights": K.bit_llr(n, hash_bits, N_CLASSES,
                                                         reveal_aggregates=False),
        "phash-sieve/hash_stability (B=256)": K.hash_stability(n, hash_bits),
        "phash-sieve/block_collision (49 tiles x 16 bit)": K.block_collision(
            N_PER_CLASS, 49, 16, N_CLASSES),
        "phash-sieve/hash_spectral (B=256)": K.hash_spectral(N_PER_CLASS, hash_bits,
                                                             10, N_CLASSES),
        "phash-sieve/hamming_ball (B=256)": K.hamming_ball(N_PER_CLASS, hash_bits, N_CLASSES),
        "phash-sieve/hamming_knn (B=256, k=10)": K.hamming_knn(N_PER_CLASS, hash_bits,
                                                               10, N_CLASSES),
        "baseline/pixel k-NN (d=3072)": K.pixel_knn(N_PER_CLASS, 3072, 10, N_CLASSES),
        "baseline/activation clustering (k-means, d=512)": K.kmeans(N_PER_CLASS, 512, 2,
                                                                    20, N_CLASSES),
        "baseline/ResNet-18 inference (50k images)": K.cnn_forward(n, macs, relus),
        "baseline/ResNet-18 training (30 epochs)": K.cnn_training(n, macs, relus, 30),
        "baseline/spectral signatures (train+infer+SVD)": K.spectral_signatures(n, macs, relus),
    }
    rows = []
    for label, circ in entries.items():
        summary = {f"circ_{k}": v for k, v in circ.summary().items()}
        for proto in PROTOCOLS:
            for net in ("LAN", "WAN"):
                r = cost_report(circ, protocol=proto, network=net, n_items=n)
                rows.append(dict(component=label, **summary, **r))
        r = cost_report(circ, protocol="3pc-replicated", network="LAN", n_items=n)
        print(f"  {label:48s} {r['online_MB']:14.1f} MB  {r['rounds']:10d} rounds  "
              f"{r['time_ms'] / 1000:12.2f} s (3PC/LAN)", flush=True)
    return pd.DataFrame(rows)


def hash_stage_cost(hash_bits=256, n=N_TRAIN, frac_bits=16, images=None):
    """Cost of hashing the whole dataset, for the total pipeline figure."""
    rows = []
    for name in ("ahash16", "rhash16", "phash", "phash_mean", "pdq"):
        c, _, _ = C.verify(name, images[:64], frac_bits=frac_bits)
        per_image = c.scaled(1)
        for attr in ("n_mult", "n_and", "n_b2a", "n_trunc", "n_local_mac"):
            setattr(per_image, attr, int(getattr(c, attr) / 64))
        per_image.n_cmp = {w: int(v / 64) for w, v in c.n_cmp.items()}
        whole = per_image.scaled(n)
        whole.rounds = per_image.rounds        # all images hash in parallel
        summary = {f"circ_{k}": v for k, v in whole.summary().items()}
        for proto in PROTOCOLS:
            for net in ("LAN", "WAN"):
                rows.append(dict(component=f"hashing/{name} (50k images)", **summary,
                                 **cost_report(whole, protocol=proto, network=net, n_items=n)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=512)
    ap.add_argument("--frac-bits", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    images = sample_images(args.n_images)

    print("== 1. correctness of the fixed-point MPC formulation ==", flush=True)
    df_c = correctness_table(images)
    df_c.to_csv(osp.join(RESULTS, "mpc_correctness.csv"), index=False)

    print("\n== 2. per-image hash cost ==", flush=True)
    df_h = hash_cost_table(images, frac_bits=args.frac_bits)
    df_h.to_csv(osp.join(RESULTS, "mpc_hash_cost.csv"), index=False)

    print("\n== 3. ResNet-18 profile ==", flush=True)
    macs, relus = resnet18_profile()
    print(f"  MACs/forward = {macs / 1e6:.1f}M, ReLU units = {relus / 1e3:.1f}k", flush=True)

    print("\n== 4. whole-pipeline cost (50 000 CIFAR-10 samples) ==", flush=True)
    df_p = pd.concat([hash_stage_cost(images=images, frac_bits=args.frac_bits),
                      pipeline_cost_table(macs, relus),
                      deployed_configurations(images, args.frac_bits)], ignore_index=True)
    df_p.to_csv(osp.join(RESULTS, "mpc_pipeline_cost.csv"), index=False)

    print("\nwrote results/mpc_correctness.csv, mpc_hash_cost.csv, mpc_pipeline_cost.csv")


if __name__ == "__main__":
    main()
