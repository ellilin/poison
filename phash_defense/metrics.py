"""Detection and purification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _safe_auc(y, s):
    y = np.asarray(y, dtype=int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def tpr_at_fpr(y, s, fpr_target=0.01):
    """Largest TPR achievable while keeping FPR <= ``fpr_target``."""
    y = np.asarray(y, dtype=int)
    if y.min() == y.max():
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def detection_metrics(poison_mask, scores, labels=None, target_class=None, keep_mask=None):
    """Full metric bundle for one (attack, hash, detector) cell.

    ``poison_mask``  : ground-truth boolean poison indicator
    ``scores``       : suspicion scores, higher = more suspicious
    ``keep_mask``    : optional boolean mask produced by the defense's own threshold
    """
    y = np.asarray(poison_mask, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    # The defender does not know the *sign* of the effect a trigger has on the
    # statistic: a shared trigger makes poisoned samples unusually typical of
    # their class, while a flipped label makes them unusually atypical, and the
    # two push the score in opposite directions.  So both the one-sided and the
    # two-sided (|z|) statistic are reported; the two-sided one is what the
    # deployed defense uses.
    a = np.abs(s)
    out = {
        "auc": _safe_auc(y, s),
        "auc_abs": _safe_auc(y, a),
        "ap": float(average_precision_score(y, s)) if y.min() != y.max() else float("nan"),
        "tpr@1fpr": tpr_at_fpr(y, s, 0.01),
        "tpr@5fpr": tpr_at_fpr(y, s, 0.05),
        "tpr@1fpr_abs": tpr_at_fpr(y, a, 0.01),
        "poison_rate": float(y.mean()),
    }

    # metrics restricted to the attacked class (the "oracle class" setting used by
    # spectral-signature style defenses, which purify one class at a time)
    if labels is not None and target_class is not None:
        m = np.asarray(labels) == target_class
        auc_t = _safe_auc(y[m], s[m])
        out["auc_target_class"] = auc_t
        out["auc_target_class_abs"] = _safe_auc(y[m], a[m])
        out["tpr@1fpr_target_class"] = tpr_at_fpr(y[m], s[m], 0.01)
        out["tpr@1fpr_target_class_abs"] = tpr_at_fpr(y[m], a[m], 0.01)
        # Two mechanisms push the score in opposite directions: a shared trigger
        # makes the poison unusually *typical* of the class (+), a flipped label
        # makes it unusually *atypical* (-). Both signs are a priori meaningful,
        # so the achievable power is max(auc, 1-auc) together with its direction.
        finite = np.isfinite(auc_t)
        out["auc_target_class_best"] = max(auc_t, 1 - auc_t) if finite else float("nan")
        out["direction"] = (1 if auc_t >= 0.5 else -1) if finite else 0
        out["tpr@1fpr_target_class_best"] = max(
            tpr_at_fpr(y[m], s[m], 0.01), tpr_at_fpr(y[m], -s[m], 0.01)) if finite \
            else float("nan")

    if keep_mask is not None:
        keep = np.asarray(keep_mask, dtype=bool)
        removed = ~keep
        n_poison = max(int(y.sum()), 1)
        n_clean = max(int((1 - y).sum()), 1)
        tp = int((removed & (y == 1)).sum())
        fp = int((removed & (y == 0)).sum())
        out.update({
            "elimination_rate": tp / n_poison,       # fraction of poison removed
            "sacrifice_rate": fp / n_clean,          # fraction of clean data lost
            "precision": tp / max(tp + fp, 1),
            "recall": tp / n_poison,
            "f1": 2 * tp / max(2 * tp + fp + (n_poison - tp), 1),
            "n_removed": int(removed.sum()),
            "residual_poison_rate": float((y[keep]).mean()) if keep.any() else 0.0,
        })
    return out


def summarise(rows, by=("attack", "hash", "detector"), metric="auc"):
    """Pivot a list of metric dicts into a table (pandas optional)."""
    import pandas as pd

    df = pd.DataFrame(rows)
    if len(by) == 3:
        return df.pivot_table(index=[by[1], by[2]], columns=by[0], values=metric)
    return df
