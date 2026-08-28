#!/usr/bin/env python
"""Figures for the report (written to results/figures/)."""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))

from phash_defense import attacks as A  # noqa: E402
from phash_defense import detectors as D  # noqa: E402
from phash_defense import hashes as H  # noqa: E402

RESULTS = osp.abspath(osp.join(osp.dirname(__file__), "..", "results"))
FIGS = osp.join(RESULTS, "figures")

plt.rcParams.update({"figure.dpi": 130, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "savefig.bbox": "tight"})

#: primary metric everywhere: AUC inside the attacked class, best of the two
#: directions (a shared trigger and a flipped label move the score opposite ways)
KEY = "auc_target_class_best"


def _save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    path = osp.join(FIGS, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def fig_auc_heatmap(df):
    best = (df.groupby(["hash", "attack"])[KEY].max().reset_index()
              .pivot(index="hash", columns="attack", values=KEY))
    order = best.mean(axis=1).sort_values(ascending=False).index
    best = best.loc[order]
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * best.shape[1], 0.9 + 0.24 * best.shape[0]))
    im = ax.imshow(best.values, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(best.shape[1]))
    ax.set_xticklabels(best.columns, rotation=45, ha="right")
    ax.set_yticks(range(best.shape[0]))
    ax.set_yticklabels(best.index)
    for i in range(best.shape[0]):
        for j in range(best.shape[1]):
            v = best.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                        color="black")
    ax.set_title("Detection ROC-AUC (best detector per cell)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="AUC")
    _save(fig, "auc_heatmap.png")


def fig_delta_heatmap(df, control_csv):
    """Detection power *above the label-flip control*: the trigger-specific part."""
    if not osp.exists(control_csv):
        return
    ctl = pd.read_csv(control_csv)
    ctl = ctl[np.isfinite(ctl[KEY])]
    base = ctl.set_index(["hash", "detector"])[KEY]
    d = df.copy()
    d["delta"] = [r[KEY] - base.get((r["hash"], r["detector"]), 0.5) for _, r in d.iterrows()]
    piv = d.pivot_table(index="hash", columns="attack", values="delta", aggfunc="max")
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * piv.shape[1], 0.9 + 0.24 * piv.shape[0]))
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0.0, vmax=0.5, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5)
    ax.set_title("Trigger-specific detection power (AUC above the label-flip control)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label=r"$\Delta$AUC")
    _save(fig, "delta_heatmap.png")


def fig_detector_comparison(df):
    piv = df.pivot_table(index="detector", columns="attack", values=KEY, aggfunc="max")
    m = piv.mean(axis=1).sort_values()
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.barh(m.index, m.values, color="#4878a8")
    ax.axvline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xlabel("mean best-AUC over attacks")
    ax.set_title("Which detection statistic works on hash bits?")
    _save(fig, "detector_comparison.png")


def fig_bits_vs_auc(df):
    g = df.groupby(["hash", "n_bits", "family"])[KEY].mean().reset_index()
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for fam, sub in g.groupby("family"):
        ax.scatter(sub["n_bits"], sub[KEY], label=fam, s=22)
        for _, r in sub.iterrows():
            ax.annotate(r["hash"], (r["n_bits"], r[KEY]), fontsize=5,
                        xytext=(2, 2), textcoords="offset points")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("hash length (bits)")
    ax.set_ylabel("mean AUC over attacks")
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.legend(fontsize=6)
    ax.set_title("Detection power vs hash length")
    _save(fig, "bits_vs_auc.png")


def fig_rate_sweep(df):
    if df["rate"].nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for attack, sub in df.groupby("attack"):
        s = sub.groupby("rate")[KEY].max()
        ax.plot(s.index * 100, s.values, marker="o", label=attack)
    ax.set_xlabel("poison rate (%)")
    ax.set_ylabel("best AUC")
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.legend(fontsize=6)
    ax.set_title("Detectability vs poison rate")
    _save(fig, "rate_sweep.png")


def fig_mpc_pareto(df, cost_csv):
    if not osp.exists(cost_csv):
        return
    cost = pd.read_csv(cost_csv)
    cost = cost[(cost["protocol"].str.startswith("3PC")) & (cost["network"] == "LAN")]
    cost = cost.set_index("hash")["online_bits"].to_dict()
    g = df.groupby("hash")[KEY].mean()
    xs, ys, names = [], [], []
    for h, auc in g.items():
        if h in cost:
            xs.append(cost[h] / 8 / 1e3)   # kB per image
            ys.append(auc)
            names.append(h)
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.scatter(xs, ys, s=24, color="#b8562f")
    for x, y, n in zip(xs, ys, names):
        ax.annotate(n, (x, y), fontsize=5.5, xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("online communication per image (kB, 3PC/LAN)")
    ax.set_ylabel("mean AUC over attacks")
    ax.set_title("MPC cost vs detection power")
    _save(fig, "mpc_pareto.png")


def fig_attack_examples(attacks, rate, y_target, seed):
    n = len(attacks)
    fig, axes = plt.subplots(2, n, figsize=(1.05 * n, 2.4))
    for k, name in enumerate(attacks):
        try:
            ps = A.build(name, poisoned_rate=rate, y_target=y_target, seed=seed)
        except Exception as exc:  # noqa: BLE001
            print(f"  (skip {name}: {exc})")
            continue
        idx = np.flatnonzero(ps.poison_mask)[0]
        from phash_defense.data import cifar10

        clean, _ = cifar10(train=True)
        axes[0, k].imshow(clean[idx])
        axes[1, k].imshow(ps.images[idx])
        axes[0, k].set_title(name, fontsize=5.5)
        for r in (0, 1):
            axes[r, k].set_xticks([])
            axes[r, k].set_yticks([])
            axes[r, k].grid(False)
    axes[0, 0].set_ylabel("clean", fontsize=6)
    axes[1, 0].set_ylabel("poisoned", fontsize=6)
    _save(fig, "attack_examples.png")


def fig_trigger_response(attack, rate, y_target, seed):
    """Which hash bits does the trigger actually flip, and where are they?"""
    from phash_defense.data import cifar10

    clean, _ = cifar10(train=True)
    ps = A.build(attack, poisoned_rate=rate, y_target=y_target, seed=seed)
    idx = np.flatnonzero(ps.poison_mask)[:2000]
    names = ["ahash", "ahash16", "dhash16", "blockhash", "phash", "rhash16", "rhash32"]
    fig, axes = plt.subplots(1, len(names), figsize=(1.25 * len(names), 1.7))
    for ax, name in zip(axes, names):
        a = H.compute(name, clean[idx])
        b = H.compute(name, ps.images[idx])
        flip = (a != b).mean(axis=0)
        spec = H.HASHES[name]
        if spec.layout is not None and spec.grid is not None:
            gh, gw = spec.grid
            img = np.zeros((gh, gw))
            cnt = np.zeros((gh, gw))
            for bit, (r, c) in enumerate(spec.layout):
                img[r, c] += flip[bit]
                cnt[r, c] += 1
            ax.imshow(img / np.maximum(cnt, 1), cmap="magma", vmin=0, vmax=1)
        else:
            side = int(np.ceil(np.sqrt(len(flip))))
            pad = np.zeros(side * side)
            pad[:len(flip)] = flip
            ax.imshow(pad.reshape(side, side), cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"{name}\n{flip.mean() * 100:.1f}% bits", fontsize=5.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle(f"Bit-flip probability induced by the {attack} trigger", fontsize=7)
    _save(fig, f"trigger_response_{attack}.png")


def fig_score_distributions(attack, rate, y_target, seed, hash_name, detector):
    ps = A.build(attack, poisoned_rate=rate, y_target=y_target, seed=seed)
    spec = H.HASHES[hash_name]
    bits = H.compute(hash_name, ps.images)
    s = D.run(detector, bits, ps.labels, layout=spec.layout)
    if s is None:
        return
    m = ps.labels == ps.y_target
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    ax.hist(s[m & ~ps.poison_mask], bins=60, alpha=0.65, label="clean (target class)",
            density=True)
    ax.hist(s[ps.poison_mask], bins=60, alpha=0.65, label="poisoned", density=True)
    ax.set_xlabel(f"{detector} score on {hash_name} (robust z)")
    ax.set_ylabel("density")
    ax.legend(fontsize=6)
    ax.set_title(f"{attack}: separation inside the attacked class")
    _save(fig, f"scores_{attack}_{hash_name}_{detector}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detection", default=osp.join(RESULTS, "detection.csv"))
    ap.add_argument("--mpc", default=osp.join(RESULTS, "mpc_hash_cost.csv"))
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--y-target", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-data", action="store_true",
                    help="only draw figures that need results/detection.csv")
    args = ap.parse_args()

    os.makedirs(FIGS, exist_ok=True)
    if osp.exists(args.detection):
        df = pd.read_csv(args.detection)
        df = df[np.isfinite(df[KEY])]
        print("== figures from the detection grid ==")
        fig_auc_heatmap(df)
        fig_delta_heatmap(df, osp.join(RESULTS, "detection_control.csv"))
        fig_detector_comparison(df)
        fig_bits_vs_auc(df)
        fig_rate_sweep(df)
        p_rates = osp.join(RESULTS, "detection_rates.csv")
        if osp.exists(p_rates):
            rdf = pd.read_csv(p_rates)
            fig_rate_sweep(rdf[np.isfinite(rdf[KEY])])
        fig_mpc_pareto(df, args.mpc)
    else:
        print(f"(no {args.detection}; skipping the detection figures)")

    if not args.skip_data:
        print("== figures that rebuild the datasets ==")
        try:
            fig_attack_examples(["badnets", "blended", "wanet", "refool", "batt",
                                 "stego", "phash_adaptive"], args.rate, args.y_target,
                                args.seed)
            for atk in ("badnets", "blended", "wanet", "phash_adaptive"):
                fig_trigger_response(atk, args.rate, args.y_target, args.seed)
            for atk, hn, dn in [("badnets", "rhash32", "block_collision"),
                                ("blended", "rhash32", "bit_llr"),
                                ("clean_label_badnets", "phash", "bit_llr"),
                                ("wanet", "rhash32", "bit_llr")]:
                fig_score_distributions(atk, args.rate, args.y_target, args.seed, hn, dn)
        except Exception as exc:  # noqa: BLE001
            print(f"  (skipped: {exc})")


if __name__ == "__main__":
    main()
