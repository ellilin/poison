"""Poison detectors that operate on the *hash matrix* of a labelled dataset.

Every detector returns a per-sample suspicion score (higher = more likely to be
poisoned).  Scores are computed inside each label class and then robustly
standardised (median / MAD), so that they are comparable across classes: an
all-to-one backdoor concentrates the poison in a single class, and the
standardisation is what lets a single global threshold flag it.

The detectors are deliberately of different computational shapes, because the
second question of the thesis is which of them survives translation to MPC:

  * ``bit_llr`` / ``bit_em``      -- O(nB), only linear scans over the bits
  * ``hamming_knn`` / ``hamming_ball`` -- O(n^2 B), pairwise Hamming distances
  * ``block_collision``          -- O(n log n) secure sort / oblivious histogram
  * ``hash_spectral``            -- O(nB * iters), power iteration
  * ``hash_stability``           -- O(n), *no* cross-sample access at all
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def pm1(bits: np.ndarray) -> np.ndarray:
    """bool (N,B) -> float32 (N,B) in {-1,+1}."""
    return np.asarray(bits, dtype=np.float32) * 2.0 - 1.0


def robust_z(x: np.ndarray) -> np.ndarray:
    """Median/MAD standardisation (constant-free, breakdown point 50%)."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = x.std() + EPS
    return (x - med) / scale


def pairwise_hamming_blocks(bits: np.ndarray, chunk: int = 1024):
    """Yield ``(i0, D)`` where ``D[r, j]`` is the Hamming distance between row
    ``i0+r`` and row ``j``.  Uses the ``d = (B - <m_i, m_j>) / 2`` identity so
    that BLAS does the work."""
    m = pm1(bits)
    B = m.shape[1]
    for i in range(0, len(m), chunk):
        g = m[i:i + chunk] @ m.T
        yield i, (B - g) * 0.5


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


@dataclass
class DetectorSpec:
    name: str
    fn: Callable
    complexity: str
    needs_layout: bool = False
    needs_images: bool = False
    notes: str = ""
    params: dict = field(default_factory=dict)


DETECTORS: Dict[str, DetectorSpec] = {}


def register(name, complexity, needs_layout=False, needs_images=False, notes="", **params):
    def deco(fn):
        DETECTORS[name] = DetectorSpec(name, fn, complexity, needs_layout, needs_images, notes, params)
        return fn

    return deco


# --------------------------------------------------------------------------- #
# 1. Linear-scan detectors over the bits
# --------------------------------------------------------------------------- #


@register("bit_llr", "O(nB)",
          notes="Two-sample bit log-likelihood ratio: the bit profile of the class "
                "under test against the bit profile of the remaining classes.")
def bit_llr(bits, labels, cls, layout=None, alpha=0.5, **kw):
    """Score = log P(h | class-c profile) - log P(h | background profile).

    A trigger shared by the poisoned samples shifts the class-conditional bit
    frequencies away from the background; samples that carry the shift score high.
    """
    inc = labels == cls
    Hc = np.asarray(bits[inc], dtype=np.float32)
    Ho = np.asarray(bits[~inc], dtype=np.float32)
    q = (Hc.sum(0) + alpha) / (len(Hc) + 2 * alpha)
    p = (Ho.sum(0) + alpha) / (len(Ho) + 2 * alpha)
    w1 = np.log(q / p)
    w0 = np.log((1 - q) / (1 - p))
    return Hc @ w1 + (1.0 - Hc) @ w0


@register("bit_em", "O(TnB)",
          notes="Two-component Bernoulli mixture fitted inside the class by EM; the "
                "low-entropy component is declared to be the poison. Gives the "
                "posterior probability of being poisoned.")
def bit_em(bits, labels, cls, layout=None, eps_prior=0.05, iters=30, alpha=0.5, seed=0, **kw):
    Hc = np.asarray(bits[labels == cls], dtype=np.float32)
    n, B = Hc.shape
    if n < 20:
        return np.zeros(n, dtype=np.float32)

    # initialise the poison component from the top-eps samples of the LLR statistic
    init = bit_llr(bits, labels, cls, alpha=alpha)
    k = max(2, int(np.ceil(eps_prior * n)))
    top = np.argsort(init)[-k:]
    r = np.zeros(n, dtype=np.float64)
    r[top] = 1.0

    pi = float(r.mean())
    theta = phi = np.full(B, 0.5)
    for _ in range(iters):
        w = r[:, None]
        theta = ((w * Hc).sum(0) + alpha) / (r.sum() + 2 * alpha)          # poison profile
        phi = (((1 - w) * Hc).sum(0) + alpha) / ((1 - r).sum() + 2 * alpha)  # clean profile
        theta = np.clip(theta, 1e-4, 1 - 1e-4)
        phi = np.clip(phi, 1e-4, 1 - 1e-4)
        ll1 = Hc @ np.log(theta) + (1 - Hc) @ np.log(1 - theta) + np.log(max(pi, 1e-9))
        ll0 = Hc @ np.log(phi) + (1 - Hc) @ np.log(1 - phi) + np.log(max(1 - pi, 1e-9))
        mx = np.maximum(ll1, ll0)
        r = np.exp(ll1 - mx) / (np.exp(ll1 - mx) + np.exp(ll0 - mx))
        pi = float(np.clip(r.mean(), 1e-4, 0.5))

    # the poison component must be the *tighter* one; otherwise EM latched onto a
    # semantic split and we return the flipped responsibility.
    ent_theta = -(theta * np.log(theta) + (1 - theta) * np.log(1 - theta)).mean()
    ent_phi = -(phi * np.log(phi) + (1 - phi) * np.log(1 - phi)).mean()
    return r.astype(np.float32) if ent_theta <= ent_phi else (1.0 - r).astype(np.float32)


# --------------------------------------------------------------------------- #
# 2. Pairwise-distance detectors
# --------------------------------------------------------------------------- #


#: One-entry cache so that hamming_knn and hamming_ball can share the (expensive)
#: pairwise distance pass. The array itself is kept in the cache so that the
#: ``is`` check is sound -- an id alone could be reused by a freshly allocated
#: array at the same address.
_PAIRWISE_CACHE = {"bits": None, "key": None, "val": None}


def _pairwise_scores(bits, labels, cls, k, tau, quantile, max_n, seed):
    key = (int(cls), int(k), tau, float(quantile), int(max_n), int(seed))
    if _PAIRWISE_CACHE["bits"] is bits and _PAIRWISE_CACHE["key"] == key:
        return _PAIRWISE_CACHE["val"]

    Hc = bits[labels == cls]
    n, B = len(Hc), bits.shape[1]
    idx = _subsample(n, max_n, seed)
    sub = Hc[idx] if idx is not None else Hc
    m_ref = pm1(sub)
    kk = min(k + 1, len(sub))

    if tau is not None:
        radius = tau * B
    else:
        probe = (B - pm1(Hc[:min(n, 256)]) @ m_ref.T) * 0.5
        radius = float(np.quantile(probe, quantile)) if probe.size else 0.0

    knn = np.empty(n, dtype=np.float32)
    ball = np.empty(n, dtype=np.float32)
    for i0, block in _blocks_against(Hc, m_ref, B):
        ball[i0:i0 + len(block)] = (block <= radius).sum(axis=1) - 1.0
        part = np.sort(np.partition(block, kk - 1, axis=1)[:, :kk], axis=1)[:, 1:kk]
        knn[i0:i0 + len(part)] = -part.mean(axis=1)
    _PAIRWISE_CACHE.update(bits=bits, key=key, val=(knn, ball))
    return knn, ball


@register("hamming_knn", "O(n^2 B)",
          notes="Mean Hamming distance to the k nearest same-class neighbours. "
                "Poisoned samples share the trigger bits, so they lie abnormally "
                "close to each other.")
def hamming_knn(bits, labels, cls, layout=None, k=10, tau=None, quantile=0.005,
                max_n=12000, seed=0, **kw):
    n = int((labels == cls).sum())
    if n <= k + 1:
        return np.zeros(n, dtype=np.float32)
    return _pairwise_scores(bits, labels, cls, k, tau, quantile, max_n, seed)[0]


@register("hamming_ball", "O(n^2 B)",
          notes="Number of same-class neighbours inside a Hamming ball. The radius is "
                "either tau*B or, by default, a low quantile of the observed "
                "within-class distance distribution (a public constant, estimated on "
                "a sample, so it costs nothing extra under MPC).")
def hamming_ball(bits, labels, cls, layout=None, k=10, tau=None, quantile=0.005,
                 max_n=12000, seed=0, **kw):
    n = int((labels == cls).sum())
    if n <= k + 1:
        return np.zeros(n, dtype=np.float32)
    return _pairwise_scores(bits, labels, cls, k, tau, quantile, max_n, seed)[1]


def _subsample(n, max_n, seed):
    if n <= max_n:
        return None
    return np.random.default_rng(seed).choice(n, max_n, replace=False)


def _blocks_against(query_bits, m_ref, B, chunk=1024):
    mq = pm1(query_bits)
    for i in range(0, len(mq), chunk):
        g = mq[i:i + chunk] @ m_ref.T
        yield i, (B - g) * 0.5


# --------------------------------------------------------------------------- #
# 3. Block-collision sieve (spatially local hashes only)
# --------------------------------------------------------------------------- #


@register("block_collision", "O(n m log n)", needs_layout=True,
          notes="Splits a spatially localised hash into overlapping tiles and counts "
                "exact collisions of the tile code inside the class. A fixed patch "
                "trigger makes every poisoned sample share one tile code exactly.")
def block_collision(bits, labels, cls, layout=None, tile=4, stride=2, **kw):
    if layout is None:
        return None
    Hc = np.asarray(bits[labels == cls], dtype=np.uint8)
    n, B = Hc.shape
    rows, cols = layout[:, 0], layout[:, 1]
    gh, gw = int(rows.max()) + 1, int(cols.max()) + 1
    cells = gh * gw
    n_groups = max(1, B // cells)  # channel groups for per-channel hashes

    best = np.zeros(n, dtype=np.float32)
    for g in range(n_groups):
        sl = slice(g * cells, (g + 1) * cells)
        sub = Hc[:, sl]
        r, c = rows[sl], cols[sl]
        for r0 in range(0, max(gh - tile + 1, 1), stride):
            for c0 in range(0, max(gw - tile + 1, 1), stride):
                sel = (r >= r0) & (r < r0 + tile) & (c >= c0) & (c < c0 + tile)
                if sel.sum() < 4:
                    continue
                codes = np.packbits(sub[:, sel], axis=1)
                _, inv, cnt = np.unique(codes, axis=0, return_inverse=True, return_counts=True)
                best = np.maximum(best, cnt[inv].astype(np.float32))
    return best


# --------------------------------------------------------------------------- #
# 4. Spectral signature in hash space
# --------------------------------------------------------------------------- #


@register("hash_spectral", "O(nB * iters)",
          notes="Tran et al.'s spectral signature computed on the +-1 hash matrix "
                "instead of deep features. The top singular direction is obtained by "
                "power iteration -- the same algorithm the MPC cost model prices, and "
                "O(B) cheaper than forming the covariance matrix.")
def hash_spectral(bits, labels, cls, layout=None, iters=20, seed=0, **kw):
    m = pm1(bits[labels == cls])
    if len(m) < 3:
        return np.zeros(len(m), dtype=np.float32)
    m = m - m.mean(axis=0, keepdims=True)
    v = np.random.default_rng(seed).normal(size=m.shape[1]).astype(np.float32)
    v /= np.linalg.norm(v) + EPS
    for _ in range(iters):
        v = m.T @ (m @ v)
        nrm = np.linalg.norm(v)
        if nrm < EPS:
            break
        v /= nrm
    return np.abs(m @ v)


# --------------------------------------------------------------------------- #
# 5. Hash-stability (per-sample, no cross-sample access)
# --------------------------------------------------------------------------- #


@register("hash_stability", "O(n)", needs_images=True,
          notes="Hamming distance between H(x) and H(T(x)) for a content-preserving "
                "transform T. A perceptual hash is invariant on natural content, so "
                "a fragile trigger shows up as instability.")
def hash_stability(bits, labels, cls, layout=None, stability=None, **kw):
    """``stability`` is precomputed by :func:`stability_scores` (needs the images)."""
    if stability is None:
        return None
    return np.asarray(stability, dtype=np.float32)[labels == cls]


def stability_scores(images, hash_name, transform="shrinkpad", pad=2, blur=3, batch=4096):
    """Hamming distance between the hash of ``x`` and the hash of ``T(x)``."""
    from . import hashes as H

    x = np.asarray(images)
    if transform == "shrinkpad":
        t = _shrink_pad(x, pad)
    elif transform == "blur":
        t = _blur_rgb(x, blur)
    elif transform == "jpegish":
        t = (x.astype(np.int16) // 8 * 8).astype(np.uint8)
    else:
        raise ValueError(transform)
    a = H.compute(hash_name, x, batch=batch)
    b = H.compute(hash_name, t, batch=batch)
    return (a != b).sum(axis=1).astype(np.float32)


def _shrink_pad(x, pad):
    """ShrinkPad (Li et al.): shrink by 2*pad and pad back at a random offset."""
    from .hashes import resize

    n, h, w, c = x.shape
    small = np.stack([resize(x[..., k].astype(np.float64), h - 2 * pad, w - 2 * pad)
                      for k in range(c)], axis=-1)
    out = np.zeros_like(x, dtype=np.float64)
    out[:, pad:h - pad, pad:w - pad, :] = small
    return np.clip(out, 0, 255).astype(np.uint8)


def _blur_rgb(x, k):
    from .hashes import box_blur

    return np.clip(np.stack([box_blur(x[..., c].astype(np.float64), k) for c in range(x.shape[-1])],
                            axis=-1), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run(name: str,
        bits: np.ndarray,
        labels: np.ndarray,
        layout: Optional[np.ndarray] = None,
        standardise: bool = True,
        **params) -> Optional[np.ndarray]:
    """Run a detector on every class and return globally comparable scores.

    Returns ``None`` when the detector does not apply to this hash (e.g. a
    block-collision sieve on a frequency-domain hash without spatial layout).
    """
    spec = DETECTORS[name]
    if spec.needs_layout and layout is None:
        return None
    out = np.zeros(len(labels), dtype=np.float32)
    for cls in np.unique(labels):
        m = labels == cls
        s = spec.fn(bits, labels, cls, layout=layout, **params)
        if s is None:
            return None
        s = np.asarray(s, dtype=np.float32)
        out[m] = robust_z(s) if standardise else s
    return out


def ensemble(score_list, method="zmean") -> np.ndarray:
    """Combine several already-standardised score vectors.

    ``zmean`` averages the robust z-scores. ``rank`` averages the ranks, which is
    more robust to a badly-scaled member but *destroys the outlier structure*:
    ranks are uniform by construction, so a subsequent median/MAD threshold can
    never fire. Use ``rank`` only with a top-fraction rule.
    """
    if method == "zmean":
        return np.mean(np.stack(score_list), axis=0).astype(np.float32)
    ranks = []
    for s in score_list:
        r = np.empty(len(s), dtype=np.float64)
        r[np.argsort(s)] = np.arange(len(s))
        ranks.append(r / max(len(s) - 1, 1))
    return np.mean(ranks, axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Thresholding / purification
# --------------------------------------------------------------------------- #


def _suspicious_cluster(crit, iters=25, max_frac=0.5, gap_k=4.0):
    """1-D 2-means on the scores; flag the high cluster if it is the minority.

    A fixed threshold in robust standard deviations cannot fire when the poison
    is a large fraction of the class, because the poison contaminates the very
    median and MAD used to normalise it (see Proposition 9 of docs/02_theory.md).
    Splitting the score distribution into two clusters makes no assumption about
    the poison rate, and it is cheap under MPC: a handful of iterations of `n`
    comparisons and two means, all linear.
    """
    s = np.asarray(crit, dtype=np.float64)
    if len(s) < 8:
        return np.zeros(len(s), dtype=bool)
    c = np.array([np.quantile(s, 0.25), np.quantile(s, 0.99)])
    a = np.zeros(len(s), dtype=int)
    for _ in range(iters):
        a = (np.abs(s - c[1]) < np.abs(s - c[0])).astype(int)
        for j in (0, 1):
            if (a == j).any():
                c[j] = s[a == j].mean()
    high = 1 if c[1] >= c[0] else 0
    flagged = a == high
    none = np.zeros(len(s), dtype=bool)
    if flagged.mean() > max_frac:          # the "high" cluster is the majority
        return none
    # significance guard: a 2-means split always exists, so only act when the two
    # modes are genuinely separated relative to the spread of the clean mode
    lo = s[~flagged]
    if lo.size < 2:
        return none
    spread = lo.std() + EPS
    if (c[high] - c[1 - high]) < gap_k * spread:
        return none
    return flagged


def threshold_mask(scores, labels=None, rule="mad", k=3.5, frac=0.05, two_sided=True,
                   max_frac=0.5):
    """Return a boolean *keep* mask under the defense's automatic threshold.

    ``mad``    : keep samples with robust z-score below ``k`` (per class if labels given)
    ``zscore`` : mean/std instead of median/MAD -- less robust, but the mean and
                 the variance are *linear* statistics, so this is the variant that
                 is cheap under MPC (a median needs an oblivious sort)
    ``topfrac``: drop the ``frac`` most extreme samples of every class

    ``two_sided``: flag anomalies in both directions. The defender does not know
    whether the trigger makes poisoned samples unusually *typical* of their class
    (a shared trigger) or unusually *atypical* (a flipped label moves an image of
    another class into this one), so both tails must be cut.
    """
    keep = np.ones(len(scores), dtype=bool)
    groups = [np.ones(len(scores), dtype=bool)] if labels is None else \
             [labels == c for c in np.unique(labels)]
    for m in groups:
        s = scores[m]
        if rule == "mad":
            z = robust_z(s)
            keep[m] = (np.abs(z) < k) if two_sided else (z < k)
        elif rule == "zscore":
            z = (s - s.mean()) / (s.std() + EPS)
            keep[m] = (np.abs(z) < k) if two_sided else (z < k)
        elif rule == "topfrac":
            crit = np.abs(robust_z(s)) if two_sided else s
            n_drop = int(np.ceil(frac * len(s)))
            if n_drop:
                order = np.argsort(crit)
                sub = np.ones(len(s), dtype=bool)
                sub[order[-n_drop:]] = False
                keep[m] = sub
        elif rule == "cluster":
            crit = np.abs(robust_z(s)) if two_sided else robust_z(s)
            keep[m] = ~_suspicious_cluster(crit, max_frac=max_frac)
        else:
            raise ValueError(rule)
    return keep
