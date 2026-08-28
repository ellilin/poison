import numpy as np
import pytest

from phash_defense import hashes as H


@pytest.fixture(scope="module")
def imgs():
    rng = np.random.default_rng(0)
    # smooth-ish images: white noise is the worst case for every resampling filter
    base = rng.integers(0, 256, (32, 8, 8, 3)).astype(np.float64)
    up = np.repeat(np.repeat(base, 4, axis=1), 4, axis=2)
    return np.clip(up + rng.normal(0, 8, up.shape), 0, 255).astype(np.uint8)


def test_every_hash_returns_the_declared_number_of_bits(imgs):
    for name, spec in H.HASHES.items():
        bits = H.compute(name, imgs)
        assert bits.dtype == bool, name
        assert bits.shape == (len(imgs), spec.n_bits), name


def test_hashes_are_deterministic(imgs):
    for name in H.HASHES:
        assert np.array_equal(H.compute(name, imgs), H.compute(name, imgs)), name


def test_layout_is_consistent_with_the_bit_count():
    for name, spec in H.HASHES.items():
        if spec.layout is not None:
            assert spec.layout.shape == (spec.n_bits, 2), name


def test_perceptual_hashes_are_robust_to_small_noise(imgs):
    """The defining property of a perceptual hash: small perturbations keep it."""
    rng = np.random.default_rng(1)
    noisy = np.clip(imgs.astype(np.int16) + rng.integers(-4, 5, imgs.shape), 0, 255)
    noisy = noisy.astype(np.uint8)
    for name in ("ahash", "phash", "whash_haar", "blockhash", "pdq"):
        a, b = H.compute(name, imgs), H.compute(name, noisy)
        flip = (a != b).mean()
        assert flip < 0.10, f"{name} flipped {flip:.3f} of its bits on +-4/255 noise"


def test_haar_wavelet_hash_equals_median_average_hash(imgs):
    """Structural claim of docs/02_theory.md: the Haar LL band is a block mean."""
    assert np.array_equal(H.compute("whash_haar", imgs), H.compute("mhash", imgs))


def test_block_mean_pooling_annihilates_the_nyquist_checkerboard(imgs):
    """The null space of a pooling hash: exactly what the adaptive attack exploits.

    Exact for hashes whose pooling is an integer-ratio block mean; only
    approximate for hashes that resample at a non-integer ratio (dhash16) or
    that subtract an edge-padded blur (rhash8/16), where the image border breaks
    the cancellation.
    """
    i, j = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    board = ((-1.0) ** (i + j))[:, :, None]
    base = np.clip(imgs.astype(np.int16), 24, 231).astype(np.uint8)
    pert = np.clip(base.astype(np.float32) + 16 * board[None], 0, 255).astype(np.uint8)
    for name in ("ahash", "ahash16", "mhash", "whash_haar", "blockhash", "dhash"):
        assert np.array_equal(H.compute(name, base), H.compute(name, pert)), name
    for name in ("dhash16", "rhash8", "rhash16", "phash", "pdq"):
        flip = (H.compute(name, base) != H.compute(name, pert)).mean()
        assert flip < 0.02, f"{name} flipped {flip:.3f}"
    # ... while the non-pooled / non-linear residual hashes do see it
    for name in ("rhash32", "rhash_energy"):
        assert (H.compute(name, base) != H.compute(name, pert)).mean() > 0.05, name


def test_lanczos_mode_reproduces_the_imagehash_reference(imgs):
    """A resampling filter is a public linear map whatever its kernel, so the
    MPC formulation can reproduce the reference hashes exactly."""
    imagehash = pytest.importorskip("imagehash")
    from PIL import Image

    refs = {
        "ahash": lambda im: imagehash.average_hash(im, 8),
        "dhash": lambda im: imagehash.dhash(im, 8),
        "whash_db4": lambda im: imagehash.whash(im, 8, mode="db4"),
    }
    for name, fn in refs.items():
        ref = np.stack([np.asarray(fn(Image.fromarray(a)).hash).ravel() for a in imgs])
        with H.resample_filter("lanczos"):
            ours = H.compute(name, imgs)
        assert np.array_equal(ref, ours), name


def test_resize_matrices_are_row_stochastic():
    for n_in, n_out in [(32, 8), (32, 9), (32, 16), (32, 64)]:
        A = H.resize_matrix(n_in, n_out)
        assert A.shape == (n_out, n_in)
        assert np.allclose(A.sum(axis=1), 1.0)


def test_dct_matrix_is_orthonormal():
    for n in (8, 16, 32, 64):
        D = H.dct_matrix(n)
        assert np.allclose(D @ D.T, np.eye(n), atol=1e-10)


def test_pack_and_hamming_round_trip(imgs):
    bits = H.compute("ahash", imgs)
    packed = H.pack(bits)
    d = H.hamming(packed, packed)
    assert np.array_equal(np.diag(d), np.zeros(len(imgs), dtype=np.int32))
    brute = (bits[:, None, :] != bits[None, :, :]).sum(-1)
    assert np.array_equal(d, brute.astype(np.int32))
