"""A zoo of perceptual hash functions, written so that every hash is expressible
as ``binarise(L x)`` where ``L`` is a *public linear map* and ``binarise`` is a
comparison against either the mean (a linear statistic) or the median (an
order statistic).

That normal form is what makes the MPC analysis in ``phash_defense.mpc``
possible: the linear part is free under linear secret sharing, and the whole
online cost of a hash is concentrated in the binarisation step.

Conventions
-----------
* input  : ``uint8`` array of shape ``(N, H, W, 3)`` (RGB) or ``(N, H, W)`` (gray)
* output : ``bool`` array of shape ``(N, B)``, ``B = n_bits``

Every hash is registered in :data:`HASHES` together with metadata describing
its MPC-relevant structure (number of bits, kind of threshold, whether the
bits carry spatial locality, ...).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Basic building blocks (all linear maps are cached and exposed for the MPC
# cost model, which needs to count non-zero coefficients).
# --------------------------------------------------------------------------- #

#: ITU-R BT.601 luma weights, identical to PIL's ``Image.convert("L")``.
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def to_gray(imgs: np.ndarray) -> np.ndarray:
    """(N,H,W,3) uint8 -> (N,H,W) float64 luma in [0, 255].

    Under the ``lanczos`` (PIL-faithful) mode the integer rounding of PIL's
    ``convert("L")`` is reproduced as well.
    """
    imgs = np.asarray(imgs)
    if imgs.ndim == 3:  # already gray
        return imgs.astype(np.float64)
    if imgs.ndim != 4 or imgs.shape[-1] != 3:
        raise ValueError(f"expected (N,H,W,3) or (N,H,W), got {imgs.shape}")
    if _RESAMPLE[-1] == "lanczos":
        return pil_luma(imgs).astype(np.float64)
    return imgs.astype(np.float64) @ LUMA


@functools.lru_cache(maxsize=None)
def area_matrix(n_in: int, n_out: int) -> np.ndarray:
    """Anti-aliased box ("area") resampling matrix of shape ``(n_out, n_in)``.

    Row ``i`` averages the input samples overlapping ``[i*s, (i+1)*s)`` with
    ``s = n_in / n_out``.  For integer ratios this is an exact block mean.
    """
    A = np.zeros((n_out, n_in), dtype=np.float64)
    s = n_in / n_out
    for i in range(n_out):
        lo, hi = i * s, (i + 1) * s
        j0, j1 = int(np.floor(lo)), int(np.ceil(hi))
        for j in range(j0, min(j1, n_in)):
            A[i, j] = min(hi, j + 1) - max(lo, j)
    A /= A.sum(axis=1, keepdims=True)
    return A


@functools.lru_cache(maxsize=None)
def bilinear_matrix(n_in: int, n_out: int) -> np.ndarray:
    """Bilinear (``align_corners=False``) resampling matrix ``(n_out, n_in)``."""
    A = np.zeros((n_out, n_in), dtype=np.float64)
    scale = n_in / n_out
    for i in range(n_out):
        src = (i + 0.5) * scale - 0.5
        src = min(max(src, 0.0), n_in - 1.0)
        lo = int(np.floor(src))
        hi = min(lo + 1, n_in - 1)
        w = src - lo
        A[i, lo] += 1.0 - w
        A[i, hi] += w
    return A


@functools.lru_cache(maxsize=None)
def lanczos_matrix(n_in: int, n_out: int, a: int = 3) -> np.ndarray:
    """PIL's ``Image.LANCZOS`` resampling matrix, coefficient for coefficient.

    Reproduced from Pillow's ``precompute_coeffs`` so that the "reference"
    perceptual hashes can be expressed as the same public linear map -- a
    resampling filter is linear whatever its kernel, so this costs nothing extra
    under secret sharing.
    """
    def sinc(x):
        return 1.0 if x == 0 else np.sin(np.pi * x) / (np.pi * x)

    def lanczos(x):
        return sinc(x) * sinc(x / a) if -a <= x < a else 0.0

    scale = n_in / n_out
    filterscale = max(scale, 1.0)
    support = a * filterscale
    A = np.zeros((n_out, n_in), dtype=np.float64)
    for i in range(n_out):
        center = (i + 0.5) * scale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), n_in)
        w = np.array([lanczos((j + 0.5 - center) / filterscale) for j in range(xmin, xmax)])
        A[i, xmin:xmax] = w / w.sum()
    return A


#: resampling kernel used by :func:`resize`; ``area`` is the MPC-native choice,
#: ``lanczos`` reproduces PIL (and therefore `imagehash`) exactly.
_RESAMPLE = ["area"]


class resample_filter:
    """Context manager switching the resampling kernel."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        _RESAMPLE.append(self.name)
        return self

    def __exit__(self, *exc):
        _RESAMPLE.pop()
        return False


def resize_matrix(n_in: int, n_out: int) -> np.ndarray:
    """Area kernel when down-sampling, bilinear when up-sampling (or LANCZOS)."""
    if _RESAMPLE[-1] == "lanczos":
        return np.eye(n_in) if n_out == n_in else lanczos_matrix(n_in, n_out)
    if n_out == n_in:
        return np.eye(n_in)
    return area_matrix(n_in, n_out) if n_out < n_in else bilinear_matrix(n_in, n_out)


def resize(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Separable resize of ``(N,H,W)`` with the public matrices above.

    Under the ``lanczos`` filter each pass is rounded back to 8 bits, exactly as
    Pillow's two-pass resize does; rounding is a local operation, so it is still
    free in MPC.
    """
    Ay = resize_matrix(x.shape[1], out_h)
    Ax = resize_matrix(x.shape[2], out_w)
    if _RESAMPLE[-1] != "lanczos":
        return np.einsum("ij,njk,lk->nil", Ay, x, Ax, optimize=True)
    y = np.clip(np.einsum("njk,lk->njl", x, Ax, optimize=True) + 0.5, 0, 255).astype(np.uint8)
    return np.clip(np.einsum("ij,njl->nil", Ay, y.astype(np.float64), optimize=True) + 0.5,
                   0, 255).astype(np.uint8).astype(np.float64)


@functools.lru_cache(maxsize=None)
def dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix ``D`` such that ``X_dct = D @ x``."""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    D = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    D[0] /= np.sqrt(2.0)
    return D


def dct2(x: np.ndarray) -> np.ndarray:
    """2-D DCT-II of ``(N,H,W)`` (orthonormal, separable)."""
    Dy = dct_matrix(x.shape[1])
    Dx = dct_matrix(x.shape[2])
    return np.einsum("ij,njk,lk->nil", Dy, x, Dx, optimize=True)


def box_blur(x: np.ndarray, k: int = 3) -> np.ndarray:
    """(N,H,W) k x k mean filter with edge (replicate) padding -- a linear map."""
    p = k // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p)), mode="edge")
    csum = np.cumsum(np.cumsum(xp, axis=1), axis=2)
    csum = np.pad(csum, ((0, 0), (1, 0), (1, 0)))
    H, W = x.shape[1], x.shape[2]
    out = (
        csum[:, k:, k:]
        - csum[:, :H, k:]
        - csum[:, k:, :W]
        + csum[:, :H, :W]
    )
    return out / (k * k)


def block_reduce(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Exact block mean; requires ``H % out_h == 0`` and ``W % out_w == 0``."""
    n, H, W = x.shape
    return x.reshape(n, out_h, H // out_h, out_w, W // out_w).mean(axis=(2, 4))


def _bits_gt_mean(v: np.ndarray) -> np.ndarray:
    """v: (N, B) -> bits (N, B), threshold = per-sample mean (a linear statistic)."""
    return v > v.mean(axis=1, keepdims=True)


def _bits_gt_median(v: np.ndarray) -> np.ndarray:
    """v: (N, B) -> bits (N, B), threshold = per-sample median (an order statistic)."""
    return v > np.median(v, axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


@dataclass
class HashSpec:
    """Metadata of a perceptual hash used by the evaluation and MPC modules."""

    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    n_bits: int
    family: str  # spatial | frequency | wavelet | colour | residual
    threshold: str  # mean | median | pairwise | zero | none
    #: (row, col) grid position of every bit, or ``None`` when bits are not
    #: spatially localised (frequency-domain hashes).  Used by the
    #: block-collision detector and by the trigger-localisation analysis.
    layout: Optional[np.ndarray] = None
    grid: Optional[tuple] = None
    notes: str = ""
    tags: tuple = field(default_factory=tuple)

    def __call__(self, imgs: np.ndarray) -> np.ndarray:
        return self.fn(imgs)


HASHES: Dict[str, HashSpec] = {}


def register(name, n_bits, family, threshold, grid=None, notes="", tags=()):
    def deco(fn):
        layout = None
        if grid is not None:
            gh, gw = grid
            rr, cc = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
            layout = np.stack([rr.ravel(), cc.ravel()], axis=1)
            if layout.shape[0] != n_bits:  # e.g. per-channel hashes
                reps = n_bits // layout.shape[0]
                layout = np.tile(layout, (reps, 1))
        HASHES[name] = HashSpec(name, fn, n_bits, family, threshold, layout, grid, notes, tags)
        return fn

    return deco


# --------------------------------------------------------------------------- #
# 1. Spatial hashes
# --------------------------------------------------------------------------- #


@register("ahash", 64, "spatial", "mean", grid=(8, 8),
          notes="Average hash: 8x8 box means thresholded by their own mean.",
          tags=("classic",))
def ahash(imgs, size=8):
    g = resize(to_gray(imgs), size, size).reshape(len(imgs), -1)
    return _bits_gt_mean(g)


@register("ahash4", 16, "spatial", "mean", grid=(4, 4),
          notes="Average hash at 4x4 granularity (16 bits).")
def ahash4(imgs):
    return ahash(imgs, size=4)


@register("ahash16", 256, "spatial", "mean", grid=(16, 16),
          notes="Average hash at 16x16 granularity (256 bits).")
def ahash16(imgs):
    return ahash(imgs, size=16)


@register("ahash32", 1024, "spatial", "mean", grid=(32, 32),
          notes="Average hash at 32x32 granularity (1024 bits, no down-sampling on CIFAR).")
def ahash32(imgs):
    return ahash(imgs, size=32)


@register("mhash", 64, "spatial", "median", grid=(8, 8),
          notes="Median hash: like ahash but thresholded by the median.",
          tags=("classic",))
def mhash(imgs, size=8):
    g = resize(to_gray(imgs), size, size).reshape(len(imgs), -1)
    return _bits_gt_median(g)


@register("ahash_rgb", 192, "spatial", "mean", grid=(8, 8),
          notes="Per-channel average hash (3 x 64 bits), keeps colour information.")
def ahash_rgb(imgs, size=8):
    imgs = np.asarray(imgs, dtype=np.float64)
    out = []
    for c in range(3):
        v = resize(imgs[..., c], size, size).reshape(len(imgs), -1)
        out.append(_bits_gt_mean(v))
    return np.concatenate(out, axis=1)


@register("dhash", 64, "spatial", "pairwise", grid=(8, 8),
          notes="Difference hash: sign of the horizontal gradient of a 9x8 thumbnail "
                "(imagehash convention: bit = right > left).",
          tags=("classic",))
def dhash(imgs, size=8):
    g = resize(to_gray(imgs), size, size + 1)
    return (g[:, :, 1:] > g[:, :, :-1]).reshape(len(imgs), -1)


@register("dhash_v", 64, "spatial", "pairwise", grid=(8, 8),
          notes="Vertical difference hash.")
def dhash_v(imgs, size=8):
    g = resize(to_gray(imgs), size + 1, size)
    return (g[:, 1:, :] > g[:, :-1, :]).reshape(len(imgs), -1)


@register("dhash16", 256, "spatial", "pairwise", grid=(16, 16),
          notes="Difference hash at 16x16 granularity (256 bits).")
def dhash16(imgs):
    return dhash(imgs, size=16)


@register("blockhash", 256, "spatial", "median", grid=(16, 16),
          notes="Blockhash.io: 16x16 block means thresholded by the median of "
                "their horizontal band (4 bands).",
          tags=("classic",))
def blockhash(imgs, bits=16, bands=4):
    g = to_gray(imgs)
    b = block_reduce(g, bits, bits)  # (N, bits, bits)
    n = len(imgs)
    rows_per_band = bits // bands
    out = np.empty((n, bits, bits), dtype=bool)
    half = 255.0 / 2.0
    for k in range(bands):
        sl = b[:, k * rows_per_band:(k + 1) * rows_per_band, :]
        flat = sl.reshape(n, -1)
        med = np.median(flat, axis=1)[:, None, None]
        # blockhash tie rule: on the median, the bit follows the overall brightness
        out[:, k * rows_per_band:(k + 1) * rows_per_band, :] = (sl > med) | (
            (np.abs(sl - med) < 1e-9) & (med > half)
        )
    return out.reshape(n, -1)


# --------------------------------------------------------------------------- #
# 2. Frequency-domain hashes
# --------------------------------------------------------------------------- #


@register("phash", 64, "frequency", "median",
          notes="pHash: 8x8 low-frequency DCT-II coefficients vs their median.",
          tags=("classic",))
def phash(imgs, hash_size=8, highfreq_factor=4):
    img_size = hash_size * highfreq_factor
    g = resize(to_gray(imgs), img_size, img_size)
    d = dct2(g)[:, :hash_size, :hash_size].reshape(len(imgs), -1)
    return _bits_gt_median(d)


@register("phash_nodc", 63, "frequency", "median",
          notes="pHash variant with the DC coefficient removed (brightness invariant).")
def phash_nodc(imgs, hash_size=8, highfreq_factor=4):
    img_size = hash_size * highfreq_factor
    g = resize(to_gray(imgs), img_size, img_size)
    d = dct2(g)[:, :hash_size, :hash_size].reshape(len(imgs), -1)[:, 1:]
    return _bits_gt_median(d)


@register("phash_mean", 64, "frequency", "mean",
          notes="MPC-friendly pHash: the median threshold is replaced by the mean "
                "of the (DC-free) low-frequency band.")
def phash_mean(imgs, hash_size=8, highfreq_factor=4):
    img_size = hash_size * highfreq_factor
    g = resize(to_gray(imgs), img_size, img_size)
    d = dct2(g)[:, :hash_size, :hash_size].reshape(len(imgs), -1)
    thr = d[:, 1:].mean(axis=1, keepdims=True)
    return d > thr


@register("phash16", 256, "frequency", "median",
          notes="pHash keeping a 16x16 DCT band (256 bits).")
def phash16(imgs):
    return phash(imgs, hash_size=16, highfreq_factor=2)


@register("pdq", 256, "frequency", "median",
          notes="PDQ-style hash: Jarosz box blur, 64x64 luma, 16x16 mid-frequency "
                "DCT band (rows/cols 1..16) vs median.",
          tags=("classic",))
def pdq(imgs):
    g = to_gray(imgs)
    g = box_blur(box_blur(g, 3), 3)  # two-pass Jarosz-style filter
    g = resize(g, 64, 64)
    d = dct2(g)[:, 1:17, 1:17].reshape(len(imgs), -1)
    return _bits_gt_median(d)


# --------------------------------------------------------------------------- #
# 3. Wavelet hashes
# --------------------------------------------------------------------------- #


@register("whash_haar", 64, "wavelet", "median", grid=(8, 8),
          notes="Haar wavelet hash (level-2 LL band vs median). Note: the Haar LL "
                "band is an exact block mean, so this hash equals `mhash` up to scale.",
          tags=("classic",))
def whash_haar(imgs, hash_size=8, image_scale=32):
    import pywt

    g = resize(to_gray(imgs), image_scale, image_scale) / 255.0
    level = int(np.log2(image_scale / hash_size))
    coeffs = pywt.wavedec2(g, "haar", level=level, axes=(1, 2))
    ll = coeffs[0].reshape(len(imgs), -1)
    return _bits_gt_median(ll)


@register("whash_db4", 169, "wavelet", "median",
          notes="Daubechies-4 wavelet hash, faithful to `imagehash.whash(mode='db4')`: "
                "the maximum-level Haar LL band is removed first, and the db4 LL band "
                "of a 32x32 image is 13x13, i.e. this hash is 169 bits wide, not 64.",
          tags=("classic",))
def whash_db4(imgs, hash_size=8, image_scale=32):
    import pywt

    g = resize(to_gray(imgs), image_scale, image_scale) / 255.0
    ll_max_level = int(np.log2(image_scale))
    level = ll_max_level - int(np.log2(hash_size))
    # remove the maximum-level Haar LL band (imagehash's remove_max_haar_ll)
    coeffs = list(pywt.wavedec2(g, "haar", level=ll_max_level, axes=(1, 2)))
    coeffs[0] = coeffs[0] * 0
    g = pywt.waverec2(coeffs, "haar", axes=(1, 2))
    ll = pywt.wavedec2(g, "db4", level=level, axes=(1, 2))[0]
    return _bits_gt_median(ll.reshape(len(imgs), -1))


@register("mh", 64, "frequency", "mean", grid=(8, 8),
          notes="Marr-Hildreth style hash: Laplacian-of-Gaussian response pooled on "
                "an 8x8 grid and thresholded by its mean.",
          tags=("classic",))
def mh(imgs, sigma=1.0, size=8):
    g = to_gray(imgs)
    # LoG = blur - blur' (difference of box blurs approximates the Marr wavelet)
    log = box_blur(g, 3) - box_blur(g, 7)
    v = block_reduce(log, size, size).reshape(len(imgs), -1)
    return _bits_gt_mean(v)


# --------------------------------------------------------------------------- #
# 4. Colour hash
# --------------------------------------------------------------------------- #


def pil_luma(imgs):
    """PIL's integer ``convert("L")``: ``(19595R + 38470G + 7471B + 0x8000) >> 16``."""
    x = np.asarray(imgs, dtype=np.int64)
    return ((x[..., 0] * 19595 + x[..., 1] * 38470 + x[..., 2] * 7471 + 0x8000)
            >> 16).astype(np.uint8)


def pil_hsv(imgs):
    """PIL's integer ``convert("HSV")`` (H, S, V as uint8)."""
    x = np.asarray(imgs, dtype=np.float64)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    maxc = x.max(axis=-1)
    minc = x.min(axis=-1)
    cr = maxc - minc
    flat = cr == 0
    crs = np.where(flat, 1.0, cr)
    rc, gc, bc = (maxc - r) / crs, (maxc - g) / crs, (maxc - b) / crs
    h = np.where(r == maxc, bc - gc, np.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
    h = np.mod(h / 6.0 + 1.0, 1.0)
    s = np.where(flat, 0.0, cr / np.where(maxc == 0, 1.0, maxc))
    hh = np.where(flat, 0, np.clip((h * 255.0).astype(np.int64), 0, 255))
    ss = np.where(flat, 0, np.clip((s * 255.0).astype(np.int64), 0, 255))
    return hh.astype(np.uint8), ss.astype(np.uint8), maxc.astype(np.uint8)


@register("colorhash", 42, "colour", "quantisation",
          notes="Colour hash, faithful to `imagehash.colorhash`: fractions of black, "
                "gray and 2x6 hue bins of the faint/bright colour pixels, each "
                "quantised to 3 bits. The only hash in the zoo that is not a "
                "threshold of a linear map -- it needs histogram counting.",
          tags=("classic",))
def colorhash(imgs, binbits=3):
    imgs = np.asarray(imgs)
    n = len(imgs)
    intensity = pil_luma(imgs).reshape(n, -1)
    h, s, _ = pil_hsv(imgs)
    h = h.reshape(n, -1)
    s = s.reshape(n, -1)

    mask_black = intensity < 256 // 8
    frac_black = mask_black.mean(axis=1)
    mask_gray = s < 256 // 3
    frac_gray = (~mask_black & mask_gray).mean(axis=1)
    mask_colors = ~mask_black & ~mask_gray
    mask_faint = mask_colors & (s < 256 * 2 // 3)
    mask_bright = mask_colors & (s > 256 * 2 // 3)
    c = np.maximum(mask_colors.sum(axis=1), 1)

    edges = np.linspace(0, 255, 7)
    faint = np.stack([((h >= edges[k]) & (h < edges[k + 1] if k < 5 else h <= edges[6])
                       & mask_faint).sum(axis=1) for k in range(6)], axis=1)
    bright = np.stack([((h >= edges[k]) & (h < edges[k + 1] if k < 5 else h <= edges[6])
                        & mask_bright).sum(axis=1) for k in range(6)], axis=1)

    maxvalue = 2 ** binbits
    values = [np.minimum(maxvalue - 1, (frac_black * maxvalue).astype(np.int64)),
              np.minimum(maxvalue - 1, (frac_gray * maxvalue).astype(np.int64))]
    for counts in list(faint.T) + list(bright.T):
        values.append(np.minimum(maxvalue - 1,
                                 (counts * maxvalue / c).astype(np.int64)))
    bits = []
    for v in values:
        for i in range(binbits):
            bits.append((v // (2 ** (binbits - i - 1)) % (2 ** (binbits - i))) > 0)
    return np.stack(bits, axis=1)


# --------------------------------------------------------------------------- #
# 5. Residual ("perceptual complement") hashes -- proposed in this work
# --------------------------------------------------------------------------- #


@register("rhash8", 64, "residual", "zero", grid=(8, 8),
          notes="Residual hash (ours): sign of the block-pooled high-frequency "
                "residual x - boxblur3(x). Targets exactly the signal a perceptual "
                "hash is designed to discard.",
          tags=("ours",))
def rhash8(imgs, size=8, k=3):
    g = to_gray(imgs)
    r = g - box_blur(g, k)
    v = block_reduce(r, size, size).reshape(len(imgs), -1)
    return v > 0


@register("rhash16", 256, "residual", "zero", grid=(16, 16),
          notes="Residual hash at 16x16 granularity (256 bits).",
          tags=("ours",))
def rhash16(imgs):
    return rhash8(imgs, size=16)


@register("rhash32", 1024, "residual", "zero", grid=(32, 32),
          notes="Residual hash at full 32x32 granularity (1024 bits).",
          tags=("ours",))
def rhash32(imgs):
    return rhash8(imgs, size=32)


@register("rhash_rgb", 768, "residual", "zero", grid=(16, 16),
          notes="Per-channel residual hash (3 x 256 bits): keeps chromatic triggers.",
          tags=("ours",))
def rhash_rgb(imgs, size=16, k=3):
    imgs = np.asarray(imgs, dtype=np.float64)
    out = []
    for c in range(3):
        r = imgs[..., c] - box_blur(imgs[..., c], k)
        out.append(block_reduce(r, size, size).reshape(len(imgs), -1) > 0)
    return np.concatenate(out, axis=1)


@register("rhash_energy", 256, "residual", "mean", grid=(16, 16),
          notes="Residual-energy hash (ours): block-pooled |x - boxblur3(x)| "
                "thresholded by its own mean; detects locally anomalous "
                "high-frequency energy (patch triggers, steganographic triggers).",
          tags=("ours",))
def rhash_energy(imgs, size=16, k=3):
    g = to_gray(imgs)
    r = np.abs(g - box_blur(g, k))
    v = block_reduce(r, size, size).reshape(len(imgs), -1)
    return _bits_gt_mean(v)


@register("ahash_res", 256, "residual", "mean", grid=(16, 16),
          notes="Hybrid (ours): 16x16 block means of the residual, thresholded by "
                "the image-level mean residual.")
def ahash_res(imgs, size=16, k=3):
    g = to_gray(imgs)
    r = g - box_blur(g, k)
    v = block_reduce(r, size, size).reshape(len(imgs), -1)
    return _bits_gt_mean(v)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def compute(name: str, imgs: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Compute a registered hash in batches; returns ``(N, B)`` bool array."""
    spec = HASHES[name]
    out = []
    for i in range(0, len(imgs), batch):
        out.append(np.asarray(spec.fn(imgs[i:i + batch]), dtype=bool))
    return np.concatenate(out, axis=0) if out else np.zeros((0, spec.n_bits), bool)


def pack(bits: np.ndarray) -> np.ndarray:
    """Pack a ``(N,B)`` bool array into ``(N, ceil(B/8))`` uint8 for fast Hamming."""
    return np.packbits(np.ascontiguousarray(bits, dtype=bool), axis=1)


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming(packed_a: np.ndarray, packed_b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between packed hash matrices -> (Na, Nb) int32."""
    x = np.bitwise_xor(packed_a[:, None, :], packed_b[None, :, :])
    return _POPCOUNT[x].sum(axis=2, dtype=np.int32)


def hamming_chunked(packed: np.ndarray, chunk: int = 512):
    """Yield ``(start, block)`` of the pairwise Hamming matrix, row-blocked."""
    for i in range(0, len(packed), chunk):
        yield i, hamming(packed[i:i + chunk], packed)


def list_hashes(tag: Optional[str] = None):
    if tag is None:
        return list(HASHES)
    return [k for k, v in HASHES.items() if tag in v.tags]
