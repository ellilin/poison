"""Analytic cost models for the dataset-level parts of the pipeline.

The per-image hash circuits in :mod:`circuits` are executed and counted exactly.
The dataset-level detectors (and the baselines they are compared against) are
too large to execute, so they are described analytically here as synthetic
:class:`~phash_defense.mpc.tracer.Circuit` objects.  Each formula is validated
against the traced counts on small inputs in ``tests/test_mpc.py``.
"""

from __future__ import annotations

import math

from .tracer import Circuit, ppa_and_gates


def _c(name, ring_bits=64, **kw):
    c = Circuit(name=name, ring_bits=ring_bits)
    for k, v in kw.items():
        if k == "n_cmp" or k == "n_eq":
            setattr(c, k, dict(v))
        else:
            setattr(c, k, int(v))
    return c


def _log2(x):
    return max(1, int(math.ceil(math.log2(max(x, 2)))))


# --------------------------------------------------------------------------- #
# detectors
# --------------------------------------------------------------------------- #


def bit_llr(n, B, n_classes=10, reveal_aggregates=True, ring_bits=64):
    """Cost of the bit log-likelihood-ratio detector over a whole dataset.

    The bits must first be moved from binary to arithmetic sharing (``n*B``
    conversions); the class/background bit frequencies are then free sums.

    ``reveal_aggregates``: the ``2B`` per-class bit frequencies are opened, so
    the log-ratio weights become public and the score is a *free* linear form.
    Each opened value is an average over thousands of records; if that leakage
    is unacceptable, set this to ``False`` and the weights are computed under
    encryption with a Newton division and a polynomial logarithm.
    """
    c = _c("bit_llr", ring_bits=ring_bits)
    c.n_b2a = n * B                    # bits -> arithmetic, once
    c.rounds = 1
    if reveal_aggregates:
        c.rounds += 1                  # open 2*B*n_classes aggregates
    else:
        # per (class, bit): 1 secure division (3 Newton steps = 6 mults, 6 rounds)
        # and 1 logarithm (degree-8 polynomial = 8 mults, 4 rounds), twice
        c.n_mult += n_classes * B * 2 * (6 + 8)
        c.rounds += 10
    # score: a public linear form over the arithmetic bits -> free
    c.n_local_mac = n * B
    # threshold: mean/variance are free sums, then one secure inverse square root
    c.n_mult += n                      # squares for the variance
    c.n_mult += 8                      # Newton inverse sqrt
    c.n_cmp[ring_bits] = c.n_cmp.get(ring_bits, 0) + n
    c.rounds += 1 + 8 + _log2(ring_bits)
    return c


def hamming_ball(n_per_class, B, n_classes=10, ring_bits=64):
    """All-pairs Hamming distance inside every class, thresholded.

    XOR is free in binary sharing; the population count of ``B`` bits costs
    about ``B`` AND gates with a full-adder tree, and the final radius test one
    small comparison.
    """
    pairs = n_per_class * (n_per_class - 1) // 2 * n_classes
    c = _c("hamming_ball", ring_bits=ring_bits)
    c.n_and = pairs * B
    w = _log2(B) + 1
    c.n_cmp = {w: pairs}
    c.rounds = _log2(B) + _log2(w) + 1
    return c


def hamming_knn(n_per_class, B, k=10, n_classes=10, ring_bits=64):
    """All-pairs distances plus an oblivious k-smallest selection per row."""
    c = hamming_ball(n_per_class, B, n_classes, ring_bits)
    c.name = "hamming_knn"
    w = _log2(B) + 1
    # k rounds of oblivious minimum extraction: n comparisons + n selects each
    sel = n_per_class * k * n_classes
    c.n_cmp[w] = c.n_cmp.get(w, 0) + n_per_class * n_per_class * k * n_classes
    c.n_mult += sel
    c.rounds += k * (_log2(w) + 1)
    return c


def block_collision(n_per_class, n_blocks, block_bits, n_classes=10, ring_bits=64):
    """Oblivious Batcher sort per block, then adjacent-equality run lengths."""
    n = n_per_class
    ce = int(n * _log2(n) * (_log2(n) + 1) / 4)
    c = _c("block_collision", ring_bits=ring_bits)
    total_blocks = n_blocks * n_classes
    c.n_cmp = {block_bits: ce * total_blocks}
    c.n_mult = ce * total_blocks
    c.n_eq = {block_bits: n * total_blocks}
    c.rounds = int(_log2(n) * (_log2(n) + 1) / 2) * (_log2(block_bits) + 1) + _log2(block_bits)
    return c


def hash_spectral(n_per_class, B, iters=10, n_classes=10, ring_bits=64):
    """Power iteration on the (secret) centred bit matrix."""
    c = _c("hash_spectral", ring_bits=ring_bits)
    c.n_b2a = n_per_class * B * n_classes
    c.n_mult = 2 * n_per_class * B * iters * n_classes + 8 * iters * n_classes
    c.n_cmp = {ring_bits: n_per_class * n_classes}
    c.rounds = 1 + iters * (2 + 8) + _log2(ring_bits)
    return c


def hash_stability(n, B, ring_bits=64):
    """Two hashes per sample plus a B-bit Hamming distance -- no cross-sample work."""
    c = _c("hash_stability", ring_bits=ring_bits)
    c.n_and = n * B
    w = _log2(B) + 1
    c.n_cmp = {w: n}
    c.rounds = _log2(B) + _log2(w)
    return c


# --------------------------------------------------------------------------- #
# baselines that a "classical" MPC defense would have to run
# --------------------------------------------------------------------------- #


def pixel_knn(n_per_class, d=3072, k=10, n_classes=10, ring_bits=64):
    """k-NN in raw pixel space: the naive MPC-compatible filtering baseline."""
    pairs = n_per_class * (n_per_class - 1) // 2 * n_classes
    c = _c("pixel_knn", ring_bits=ring_bits)
    c.n_mult = pairs * d                       # squared euclidean distances
    c.n_cmp = {ring_bits: n_per_class * n_per_class * k * n_classes}
    c.n_mult += n_per_class * k * n_classes
    c.rounds = 1 + k * (_log2(ring_bits) + 1)
    return c


def kmeans(n_per_class, d=512, k=2, iters=20, n_classes=10, ring_bits=64):
    """Activation clustering: k-means on penultimate features (features assumed given)."""
    c = _c("kmeans", ring_bits=ring_bits)
    n = n_per_class * n_classes
    c.n_mult = iters * n * k * d               # distances
    c.n_cmp = {ring_bits: iters * n * (k - 1)}  # argmin
    c.n_mult += iters * k * d * 8              # centroid update: secure division
    c.rounds = iters * (1 + _log2(ring_bits) + 8)
    return c


def cnn_forward(n, macs, relus, ring_bits=64):
    """One forward pass per sample with *secret* weights and secret activations."""
    c = _c("cnn_forward", ring_bits=ring_bits)
    c.n_mult = n * macs
    c.n_cmp = {ring_bits: n * relus}
    c.n_mult += n * relus                      # ReLU = comparison bit x value
    c.rounds = 60                              # ~ depth of ResNet-18 layers
    return c


def cnn_training(n, macs, relus, epochs=30, ring_bits=64):
    """Training a model under MPC: forward + backward ~ 3x forward, per epoch."""
    c = cnn_forward(n * epochs, macs, relus, ring_bits)
    c.n_mult *= 3
    for w in list(c.n_cmp):
        c.n_cmp[w] *= 3
    c.name = "cnn_training"
    c.rounds = 60 * 3 * epochs * max(1, n // 128)
    return c


def spectral_signatures(n, macs, relus, d=512, iters=10, ring_bits=64):
    """Tran et al.: train a model, extract features, run an SVD -- all under MPC."""
    c = cnn_training(n, macs, relus)
    c.merge(cnn_forward(n, macs, relus))
    c.merge(hash_spectral(n // 10, d, iters=iters))
    c.name = "spectral_signatures"
    return c


DETECTOR_COSTS = {
    "bit_llr": bit_llr,
    "hamming_ball": hamming_ball,
    "hamming_knn": hamming_knn,
    "block_collision": block_collision,
    "hash_spectral": hash_spectral,
    "hash_stability": hash_stability,
}

BASELINE_COSTS = {
    "pixel_knn": pixel_knn,
    "kmeans_activation_clustering": kmeans,
    "cnn_forward": cnn_forward,
    "cnn_training": cnn_training,
    "spectral_signatures": spectral_signatures,
}
