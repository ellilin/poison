"""MPC formulations of the perceptual hashes.

Each function takes a :class:`~phash_defense.mpc.tracer.Shared` batch of images
of shape ``(n, H, W, 3)`` and returns the hash bits as binary shares, while the
enclosing :func:`~phash_defense.mpc.tracer.trace` context accumulates the exact
primitive counts.

The point of writing them out is that the *whole* linear part of every hash --
colour conversion, resizing, DCT, wavelet transform, box blur, block pooling --
is a public linear map and therefore free under linear secret sharing.  What
remains is the binarisation, and its cost differs by two orders of magnitude
depending on which threshold the hash uses.
"""

from __future__ import annotations

import numpy as np

from .. import hashes as H
from .tracer import Shared, SharedBit, current_circuit

# --------------------------------------------------------------------------- #
# linear (free) building blocks
# --------------------------------------------------------------------------- #


def _transpose(x: Shared, axes):
    return Shared(x.v.transpose(*axes), x.depth, x.scale)


def lin2d(x: Shared, Ay, Ax) -> Shared:
    """Apply the separable public map ``Ay @ X @ Ax.T`` to every image."""
    y = x.matmul_public(Ax)                 # (n, H, Wout)
    y = _transpose(y, (0, 2, 1))            # (n, Wout, H)
    y = y.matmul_public(Ay)                 # (n, Wout, Hout)
    return _transpose(y, (0, 2, 1))         # (n, Hout, Wout)


def gray(x: Shared) -> Shared:
    """Public 3x1 colour matrix -- free."""
    return x.matmul_public(H.LUMA[None, :]).reshape(*x.shape[:3])


def resize(x: Shared, out_h, out_w) -> Shared:
    return lin2d(x, H.resize_matrix(x.shape[1], out_h), H.resize_matrix(x.shape[2], out_w))


def dct2(x: Shared) -> Shared:
    return lin2d(x, H.dct_matrix(x.shape[1]), H.dct_matrix(x.shape[2]))


def box_blur(x: Shared, k=3) -> Shared:
    """k x k mean filter written as a public (H x H) band matrix on each axis."""
    def band(n):
        A = np.zeros((n, n))
        p = k // 2
        for i in range(n):
            for d in range(-p, p + 1):
                A[i, min(max(i + d, 0), n - 1)] += 1.0 / k
        return A

    return lin2d(x, band(x.shape[1]), band(x.shape[2]))


def block_pool(x: Shared, out_h, out_w) -> Shared:
    return lin2d(x, H.area_matrix(x.shape[1], out_h), H.area_matrix(x.shape[2], out_w))


# --------------------------------------------------------------------------- #
# binarisation -- this is where all the communication happens
# --------------------------------------------------------------------------- #


def binarise_zero(v: Shared, width=None) -> SharedBit:
    """``[v_j > 0]``.  B comparisons, depth 1.  The cheapest possible threshold."""
    return (-v).ltz(width=width)


def binarise_mean(v: Shared, width=None) -> SharedBit:
    """``[v_j > mean(v)]`` rewritten as ``[B*v_j - sum(v) > 0]``.

    B comparisons, depth 1: the mean is a *linear* statistic, so it is free.
    """
    B = v.shape[-1]
    total = v.sum(axis=-1, keepdims=True)
    return (total - v.mul_public(B)).ltz(width=width)


def binarise_pairwise(v: Shared, other: Shared, width=None) -> SharedBit:
    """``[v_j > other_j]`` (used by dHash).  B comparisons, depth 1."""
    return (other - v).ltz(width=width)


def binarise_median_rank(v: Shared, width=None) -> SharedBit:
    """``[v_j > median(v)]`` via ranks.

    ``rank_j = #{i : v_i < v_j}`` needs ``B(B-1)/2`` comparisons -- but they are
    all independent, so the depth stays at two comparison layers.  This is the
    round-optimal way to threshold by a median in MPC.
    """
    B = v.shape[-1]
    a = Shared(v.v[..., :, None], v.depth, v.scale)       # (..., B, 1)
    b = Shared(v.v[..., None, :], v.depth, v.scale)       # (..., 1, B)
    # strict upper triangle only: the rest follows by antisymmetry
    iu = np.triu_indices(B, k=1)
    diff = Shared((a.v - b.v)[..., iu[0], iu[1]], v.depth, v.scale)
    lt = diff.ltz(width=width)                            # [v_i < v_j] for i<j
    # the ranks are small integers, so they are accumulated with no fractional
    # bits: the final comparison then only needs log2(B)+1 bits instead of 64
    arith = lt.to_arith(scale=0)
    full = np.zeros(v.v.shape[:-1] + (B, B), dtype=np.int64)
    full[..., iu[0], iu[1]] = arith.v
    full[..., iu[1], iu[0]] = 1 - arith.v
    rank = Shared(full.sum(axis=-2), arith.depth, 0)      # (..., B), rank_j = #{i: v_i < v_j}
    half = Shared(np.full(rank.v.shape, B // 2 - 1), 0, 0)
    w = int(np.ceil(np.log2(B))) + 1
    return (half - rank).ltz(width=w)


def binarise_median_sort(v: Shared, width=None) -> SharedBit:
    """``[v_j > median(v)]`` via a Batcher odd-even mergesort.

    Kept for comparison with :func:`binarise_median_rank`: fewer gates, but the
    sorting network is deep, which is what matters over a WAN.
    """
    B = v.shape[-1]
    idx = list(range(B))
    net = _batcher_network(B)
    x = Shared(v.v.copy(), v.depth, v.scale)
    for layer in net:
        for i, j in layer:
            a = Shared(x.v[..., i], x.depth, x.scale)
            b = Shared(x.v[..., j], x.depth, x.scale)
            c = (b - a).ltz(width=width).to_arith()        # [a > b]
            lo = b + (a - b) * c                           # min -> 1 mult
            hi = a + b - lo
            x.v[..., i] = lo.v
            x.v[..., j] = hi.v
            x.depth = max(x.depth, hi.depth)
    # even-length median: the average of the two central order statistics (free)
    med = Shared((x.v[..., B // 2 - 1:B // 2] + x.v[..., B // 2:B // 2 + 1]) // 2,
                 x.depth, x.scale)
    return (med - v).ltz(width=width)


def _batcher_network(n):
    """Layers of compare-exchange pairs of Batcher's odd-even mergesort."""
    pairs = []
    p = 1
    while p < n:
        k = p
        while k >= 1:
            layer = []
            for j in range(k % p, n - k, 2 * k):
                for i in range(min(k, n - j - k)):
                    if (i + j) // (p * 2) == (i + j + k) // (p * 2):
                        layer.append((i + j, i + j + k))
            if layer:
                pairs.append(layer)
            k //= 2
        p *= 2
    return pairs


def secure_abs(v: Shared, width=None) -> Shared:
    """``|v|`` -- one comparison and one multiplication per element.

    This is the price of a *non-linear* pooling operator; it is what makes the
    residual-energy hash strictly more expensive than every linear-pooling hash,
    and also what makes it immune to the null-space adaptive attack.
    """
    s = (-v).ltz(width=width).to_arith()          # [v > 0]
    two_s_minus_1 = s.mul_public(2) - 1.0
    return v * two_s_minus_1


# --------------------------------------------------------------------------- #
# hash circuits
# --------------------------------------------------------------------------- #


def mpc_ahash(x: Shared, size=8, width=None) -> SharedBit:
    g = resize(gray(x), size, size)
    return binarise_mean(g.reshape(len(g.v), -1), width=width)


def mpc_mhash(x: Shared, size=8, width=None, median="rank") -> SharedBit:
    g = resize(gray(x), size, size).reshape(len(x.v), -1)
    fn = binarise_median_rank if median == "rank" else binarise_median_sort
    return fn(g, width=width)


def mpc_dhash(x: Shared, size=8, width=None) -> SharedBit:
    g = resize(gray(x), size, size + 1)
    left = Shared(g.v[:, :, :-1].reshape(len(x.v), -1), g.depth, g.scale)
    right = Shared(g.v[:, :, 1:].reshape(len(x.v), -1), g.depth, g.scale)
    return binarise_pairwise(right, left, width=width)   # bit = [right > left]


def mpc_phash(x: Shared, hash_size=8, highfreq_factor=4, width=None, median="rank") -> SharedBit:
    img = hash_size * highfreq_factor
    d = dct2(resize(gray(x), img, img))
    band = Shared(d.v[:, :hash_size, :hash_size].reshape(len(x.v), -1), d.depth, d.scale)
    fn = binarise_median_rank if median == "rank" else binarise_median_sort
    return fn(band, width=width)


def mpc_phash_mean(x: Shared, hash_size=8, highfreq_factor=4, width=None) -> SharedBit:
    img = hash_size * highfreq_factor
    d = dct2(resize(gray(x), img, img))
    band = Shared(d.v[:, :hash_size, :hash_size].reshape(len(x.v), -1), d.depth, d.scale)
    ac = Shared(band.v[:, 1:], band.depth, band.scale)
    total = ac.sum(axis=-1, keepdims=True)
    return (total - band.mul_public(ac.shape[-1])).ltz(width=width)


def mpc_whash_haar(x: Shared, hash_size=8, image_scale=32, width=None, median="rank") -> SharedBit:
    """The Haar LL band is an exact block mean, hence a public linear map."""
    g = block_pool(resize(gray(x), image_scale, image_scale), hash_size, hash_size)
    fn = binarise_median_rank if median == "rank" else binarise_median_sort
    return fn(g.reshape(len(x.v), -1), width=width)


def mpc_blockhash(x: Shared, bits=16, bands=4, width=None) -> SharedBit:
    b = block_pool(gray(x), bits, bits)
    rows = bits // bands
    out = []
    for k in range(bands):
        band = Shared(b.v[:, k * rows:(k + 1) * rows, :].reshape(len(x.v), -1), b.depth, b.scale)
        out.append(binarise_median_rank(band, width=width))
    return SharedBit(np.concatenate([o.v for o in out], axis=1), max(o.depth for o in out))


def mpc_pdq(x: Shared, width=None) -> SharedBit:
    g = box_blur(box_blur(gray(x), 3), 3)
    d = dct2(resize(g, 64, 64))
    band = Shared(d.v[:, 1:17, 1:17].reshape(len(x.v), -1), d.depth, d.scale)
    return binarise_median_rank(band, width=width)


def mpc_rhash(x: Shared, size=8, k=3, width=None) -> SharedBit:
    g = gray(x)
    r = g - box_blur(g, k)
    return binarise_zero(block_pool(r, size, size).reshape(len(x.v), -1), width=width)


def mpc_rhash_energy(x: Shared, size=16, k=3, width=None) -> SharedBit:
    g = gray(x)
    r = secure_abs(g - box_blur(g, k), width=width)
    return binarise_mean(block_pool(r, size, size).reshape(len(x.v), -1), width=width)


#: name -> (circuit, plaintext hash name) for the correctness gate
MPC_HASHES = {
    "ahash": (lambda x, **kw: mpc_ahash(x, 8, **kw), "ahash"),
    "ahash16": (lambda x, **kw: mpc_ahash(x, 16, **kw), "ahash16"),
    "mhash": (lambda x, **kw: mpc_mhash(x, 8, **kw), "mhash"),
    "dhash": (lambda x, **kw: mpc_dhash(x, 8, **kw), "dhash"),
    "dhash16": (lambda x, **kw: mpc_dhash(x, 16, **kw), "dhash16"),
    "phash": (mpc_phash, "phash"),
    "phash_mean": (mpc_phash_mean, "phash_mean"),
    "whash_haar": (mpc_whash_haar, "whash_haar"),
    "blockhash": (mpc_blockhash, "blockhash"),
    "pdq": (mpc_pdq, "pdq"),
    "rhash8": (lambda x, **kw: mpc_rhash(x, 8, **kw), "rhash8"),
    "rhash16": (lambda x, **kw: mpc_rhash(x, 16, **kw), "rhash16"),
    "rhash32": (lambda x, **kw: mpc_rhash(x, 32, **kw), "rhash32"),
    "rhash_energy": (mpc_rhash_energy, "rhash_energy"),
}


def verify(name, images, frac_bits=16, **kw):
    """Run the MPC circuit and compare its bits with the plaintext hash.

    Returns ``(circuit, bit_agreement, exact_match_rate)``.
    """
    from .tracer import trace

    fn, plain = MPC_HASHES[name]
    ref = H.compute(plain, images)
    with trace(name=name, frac_bits=frac_bits) as c:
        x = Shared.encode(images.astype(np.float64))
        bits = fn(x, **kw)
        got = bits.v.astype(bool).reshape(len(images), -1)
    agree = float((got == ref).mean())
    exact = float((got == ref).all(axis=1).mean())
    return c, agree, exact
