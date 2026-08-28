#!/usr/bin/env python
"""Validate the MPC-compatible hash implementations against reference ones.

Our hashes are formulated as ``binarise(L x)`` with an explicit public linear
map ``L`` (area/bilinear resampling matrices, DCT matrices, box-blur band
matrices) because that formulation is what can be evaluated for free under
linear secret sharing.  The reference library ``imagehash`` uses PIL's
resampling filters instead.  This script measures how far apart the two are.

Also checks two structural claims made in docs/02_theory.md:
  * the Haar wavelet hash equals the median-thresholded average hash;
  * the Nyquist checkerboard perturbation leaves every mean-pooling hash
    unchanged (the null-space adaptive attack).
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

RESULTS = osp.abspath(osp.join(osp.dirname(__file__), "..", "results"))


def load_images(n):
    try:
        import torchvision

        from phash_defense.data import DATA_DIR

        ds = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=False)
        return np.asarray(ds.data[:n], dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        print(f"(CIFAR-10 unavailable: {exc}; using random images)")
        return np.random.default_rng(0).integers(0, 256, (n, 32, 32, 3), dtype=np.uint8)


def reference_bits(images, kind):
    import imagehash
    from PIL import Image

    fns = {
        "ahash": lambda im: imagehash.average_hash(im, hash_size=8),
        "dhash": lambda im: imagehash.dhash(im, hash_size=8),
        "phash": lambda im: imagehash.phash(im, hash_size=8, highfreq_factor=4),
        "whash_haar": lambda im: imagehash.whash(im, hash_size=8, mode="haar"),
        "whash_db4": lambda im: imagehash.whash(im, hash_size=8, mode="db4"),
        "colorhash": lambda im: imagehash.colorhash(im, binbits=3),
    }
    out = []
    for arr in images:
        h = fns[kind](Image.fromarray(arr))
        out.append(np.asarray(h.hash).ravel())
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default=osp.join(RESULTS, "hash_validation.csv"))
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    images = load_images(args.n)
    rows = []

    print("== agreement with the `imagehash` reference implementation ==")
    print("   (area   = the MPC-native box/bilinear resampling used everywhere else)")
    print("   (lanczos= PIL's own kernel, expressed as the same public linear map)")
    for name in ["ahash", "dhash", "phash", "whash_haar", "whash_db4", "colorhash"]:
        try:
            ref = reference_bits(images, name)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: reference unavailable ({exc})")
            continue
        line = f"  {name:12s}"
        for mode in ("area", "lanczos"):
            with H.resample_filter(mode):
                ours = H.compute(name, images)
            b = min(ref.shape[1], ours.shape[1])
            agree = float((ref[:, :b] == ours[:, :b]).mean())
            ham = (ref[:, :b] != ours[:, :b]).sum(1)
            rows.append(dict(check="imagehash_agreement", hash=name, resample=mode,
                             n_bits=b, bit_agreement=agree,
                             mean_hamming=float(ham.mean()),
                             median_hamming=float(np.median(ham)),
                             exact_match=float((ham == 0).mean())))
            line += f"  {mode}: agree={agree:.5f} exact={float((ham == 0).mean()):.3f}"
        print(line)

    print("\n== structural claim 1: Haar wavelet hash == median-thresholded average hash ==")
    a = H.compute("whash_haar", images)
    b = H.compute("mhash", images)
    agree = float((a == b).mean())
    rows.append(dict(check="whash_haar_equals_mhash", hash="whash_haar vs mhash",
                     n_bits=64, bit_agreement=agree,
                     exact_match=float((a == b).all(1).mean())))
    print(f"  bit-agreement={agree:.6f} exact={float((a == b).all(1).mean()):.4f}")

    print("\n== structural claim 2: the Nyquist checkerboard is invisible to pooling hashes ==")
    i, j = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    board = ((-1.0) ** (i + j))[:, :, None]
    # keep away from clipping so that the perturbation is exactly +-amp
    base = np.clip(images.astype(np.int16), 20, 235).astype(np.uint8)
    pert = np.clip(base.astype(np.float32) + 16 * board[None], 0, 255).astype(np.uint8)
    for name in ["ahash", "ahash16", "mhash", "phash", "phash_mean", "whash_haar",
                 "blockhash", "dhash", "dhash16", "pdq",
                 "rhash8", "rhash16", "rhash32", "rhash_energy"]:
        h0 = H.compute(name, base)
        h1 = H.compute(name, pert)
        flip = float((h0 != h1).mean())
        rows.append(dict(check="checkerboard_null_space", hash=name,
                         n_bits=int(h0.shape[1]), bit_flip_rate=flip,
                         exact_match=float((h0 == h1).all(1).mean())))
        print(f"  {name:14s} bits flipped by the trigger: {flip * 100:6.2f}%")

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
