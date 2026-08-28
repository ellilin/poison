"""Poisoned-dataset generation.

Attacks marked ``[BackdoorBox]`` are produced by the *reference implementations*
in ``BackdoorBox/core/attacks``; the remaining ones are implemented here because
they either need no model (clean-label, steganographic) or do not exist upstream
(the adaptive attacks against our own defense).

Every builder returns a :class:`~phash_defense.data.PoisonedSet` of ``uint8``
images -- i.e. exactly what a data contributor would upload and what the
defender would hold in secret-shared form.
"""

from __future__ import annotations

import os
import os.path as osp
import sys
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

from .data import CACHE_DIR, PoisonedSet, cifar10, cifar10_dataset, materialise

BBX_DIR = osp.abspath(osp.join(osp.dirname(__file__), "..", "BackdoorBox"))


def _core():
    """Import BackdoorBox's ``core`` package (kept lazy: it imports torch)."""
    if BBX_DIR not in sys.path:
        sys.path.insert(0, BBX_DIR)
    import core

    return core


def _dummy_model():
    import torch.nn as nn

    return nn.Identity(), nn.CrossEntropyLoss()


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


@dataclass
class AttackSpec:
    name: str
    builder: Callable
    source: str          # "BackdoorBox" | "this work"
    trigger_type: str    # patch | global | warping | reflection | geometric | steganographic
    visibility: str      # visible | invisible
    label: str           # dirty-label | clean-label
    notes: str = ""


ATTACKS: Dict[str, AttackSpec] = {}


def register(name, source, trigger_type, visibility, label, notes=""):
    def deco(fn):
        ATTACKS[name] = AttackSpec(name, fn, source, trigger_type, visibility, label, notes)
        return fn

    return deco


# --------------------------------------------------------------------------- #
# BackdoorBox-driven attacks
# --------------------------------------------------------------------------- #


def _bbx(make_attack, split, seed):
    """Materialise a BackdoorBox attack into arrays."""
    trainset = cifar10_dataset(train=True)
    testset = cifar10_dataset(train=False)
    atk = make_attack(trainset, testset)
    ptrain, ptest = atk.get_poisoned_dataset()
    ds = ptrain if split == "train" else ptest
    imgs, labels = materialise(ds)
    _, true_labels = cifar10(train=(split == "train"))
    mask = np.zeros(len(imgs), dtype=bool)
    mask[list(ds.poisoned_set)] = True
    return imgs, labels, true_labels, mask


@register("badnets", "BackdoorBox", "patch", "visible", "dirty-label",
          notes="3x3 white patch in the bottom-right corner (Gu et al., 2019).")
def _badnets(split, rate, y_target, seed, **kw):
    core = _core()
    model, loss = _dummy_model()
    return _bbx(lambda tr, te: core.BadNets(
        train_dataset=tr, test_dataset=te, model=model, loss=loss,
        y_target=y_target, poisoned_rate=rate, pattern=None, weight=None,
        seed=seed, deterministic=False), split, seed)


@register("badnets_1px", "BackdoorBox", "patch", "invisible", "dirty-label",
          notes="Single-pixel BadNets trigger: the weakest possible patch signal.")
def _badnets_1px(split, rate, y_target, seed, **kw):
    import torch

    core = _core()
    model, loss = _dummy_model()
    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-2, -2] = 255
    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[-2, -2] = 1.0
    return _bbx(lambda tr, te: core.BadNets(
        train_dataset=tr, test_dataset=te, model=model, loss=loss,
        y_target=y_target, poisoned_rate=rate, pattern=pattern, weight=weight,
        seed=seed, deterministic=False), split, seed)


def _blended_pattern(seed=1234):
    """The canonical 'random noise' blended trigger of Chen et al. (2017)."""
    import torch

    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.integers(0, 256, (3, 32, 32)).astype(np.uint8))


def _blended(split, rate, y_target, seed, alpha=0.1, **kw):
    import torch

    core = _core()
    model, loss = _dummy_model()
    pattern = _blended_pattern()
    weight = torch.full((3, 32, 32), float(alpha), dtype=torch.float32)
    return _bbx(lambda tr, te: core.Blended(
        train_dataset=tr, test_dataset=te, model=model, loss=loss,
        y_target=y_target, poisoned_rate=rate, pattern=pattern, weight=weight,
        seed=seed, deterministic=False), split, seed)


register("blended", "BackdoorBox", "global", "visible", "dirty-label",
         notes="Whole-image blended random-noise trigger, alpha=0.10 (Chen et al., 2017).")(
    lambda split, rate, y_target, seed, **kw: _blended(split, rate, y_target, seed, alpha=0.10))
register("blended_a05", "BackdoorBox", "global", "invisible", "dirty-label",
         notes="Blended trigger at alpha=0.05: perturbation below the perceptual threshold.")(
    lambda split, rate, y_target, seed, **kw: _blended(split, rate, y_target, seed, alpha=0.05))
register("blended_a02", "BackdoorBox", "global", "invisible", "dirty-label",
         notes="Blended trigger at alpha=0.02: the invisibility limit of the attack.")(
    lambda split, rate, y_target, seed, **kw: _blended(split, rate, y_target, seed, alpha=0.02))


@register("wanet", "BackdoorBox", "warping", "invisible", "dirty-label",
          notes="Elastic image warping (Nguyen & Tran, ICLR 2021). Sample-agnostic "
                "field, but the pixel-level effect depends on the image content.")
def _wanet(split, rate, y_target, seed, k=4, s=0.5, **kw):
    import torch
    import torch.nn as nn

    core = _core()
    model, loss = _dummy_model()
    g = torch.Generator().manual_seed(seed)
    ins = torch.rand(1, 2, k, k, generator=g) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = nn.functional.interpolate(ins, size=32, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)
    a1d = torch.linspace(-1, 1, steps=32)
    x, y = torch.meshgrid(a1d, a1d, indexing="ij")
    identity_grid = torch.stack((y, x), 2)[None, ...]
    return _bbx(lambda tr, te: core.WaNet(
        train_dataset=tr, test_dataset=te, model=model, loss=loss,
        y_target=y_target, poisoned_rate=rate, identity_grid=identity_grid,
        noise_grid=noise_grid, noise=False, seed=seed, deterministic=False), split, seed)


@register("refool", "BackdoorBox", "reflection", "visible", "dirty-label",
          notes="Reflection backdoor (Liu et al., ECCV 2020). Sample-specific natural "
                "trigger. Reflection candidates are held-out CIFAR-10 test images "
                "(the original work uses VOC2012).")
def _refool(split, rate, y_target, seed, n_candidates=200, **kw):
    core = _core()
    model, loss = _dummy_model()
    imgs_te, _ = cifar10(train=False)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(imgs_te), n_candidates, replace=False)
    cands = [imgs_te[i][:, :, ::-1].copy() for i in idx]  # BGR, as cv2 would load them
    return _bbx(lambda tr, te: core.Refool(
        train_dataset=tr, test_dataset=te, model=model, loss=loss,
        y_target=y_target, poisoned_rate=rate, reflection_candidates=cands,
        max_image_size=32, ghost_rate=0.49, seed=seed, deterministic=False), split, seed)


@register("adaptive_patch", "BackdoorBox", "patch", "visible", "dirty-label",
          notes="Adaptive attack of Qi et al. (ICLR 2023): several low-opacity patches "
                "plus 'cover' samples that carry the trigger with a correct label, "
                "explicitly designed to break latent-separability defenses.")
def _adaptive_patch(split, rate, y_target, seed, covered_rate=0.01, n_patterns=4, **kw):
    import torch

    core = _core()
    model, loss = _dummy_model()
    rng = np.random.default_rng(seed)
    patterns, alphas = [], []
    corners = [(0, 0), (0, 29), (29, 0), (29, 29)]
    for i in range(n_patterns):
        p = torch.zeros((3, 32, 32), dtype=torch.uint8)
        r0, c0 = corners[i % 4]
        col = rng.integers(64, 256, size=3)
        for ch in range(3):
            p[ch, r0:r0 + 3, c0:c0 + 3] = int(col[ch])
        patterns.append(p)
        alphas.append(0.5)
    return _bbx(lambda tr, te: core.AdaptivePatch(
        train_dataset=tr, test_dataset=te, model=model, loss=loss,
        y_target=y_target, poisoned_rate=rate, covered_rate=covered_rate,
        patterns=patterns, alphas=alphas, seed=seed, deterministic=False), split, seed)


# --------------------------------------------------------------------------- #
# Native attacks
# --------------------------------------------------------------------------- #


def _select(n, rate, rng, labels=None, y_target=None, clean_label=False):
    """Poison-index selection (matching BackdoorBox: a uniform random subset)."""
    if clean_label:
        pool = np.flatnonzero(labels == y_target)
        k = min(int(n * rate), len(pool))
        return rng.choice(pool, k, replace=False)
    k = int(n * rate)
    return rng.choice(n, k, replace=False)


def _native(split, rate, y_target, seed, apply_fn, clean_label=False):
    imgs, labels = cifar10(train=(split == "train"))
    true_labels = labels.copy()
    rng = np.random.default_rng(seed)
    if split == "test":
        idx = np.flatnonzero(labels != y_target) if not clean_label else np.arange(len(imgs))
    else:
        idx = _select(len(imgs), rate, rng, labels, y_target, clean_label)
    out = imgs.copy()
    out[idx] = apply_fn(imgs[idx], idx, rng)
    new_labels = labels.copy()
    # At test time the trigger is always expected to produce the target label --
    # that is what the attack success rate measures, clean-label or not.
    if not clean_label or split == "test":
        new_labels[idx] = y_target
    mask = np.zeros(len(imgs), dtype=bool)
    mask[idx] = True
    return out, new_labels, true_labels, mask


def _patch_white(x, size=3):
    y = x.copy()
    y[:, -size:, -size:, :] = 255
    return y


@register("label_flip_only", "this work", "none", "invisible", "dirty-label",
          notes="CONTROL, not an attack: labels are flipped to the target class with no "
                "trigger at all. Any detector scores above chance here purely because a "
                "relabelled image is atypical for its new class, so this run measures "
                "the label-noise baseline that every dirty-label number must be read "
                "against.")
def _label_flip_only(split, rate, y_target, seed, **kw):
    return _native(split, rate, y_target, seed, lambda x, idx, rng: x)


@register("clean_label_badnets", "this work", "patch", "visible", "clean-label",
          notes="BadNets patch applied only to images that already carry the target "
                "label: no label is flipped, so label-noise defenses are blind to it.")
def _clean_label(split, rate, y_target, seed, **kw):
    return _native(split, rate, y_target, seed,
                   lambda x, idx, rng: _patch_white(x), clean_label=True)


@register("batt", "this work", "geometric", "visible", "dirty-label",
          notes="BATT (Xu et al., ICASSP 2023): the trigger is a fixed 16-degree "
                "rotation. Re-implemented here because the upstream class mixes PIL "
                "and tensor types in a way that cannot be materialised.")
def _batt(split, rate, y_target, seed, angle=16, **kw):
    from PIL import Image

    def apply(x, idx, rng):
        return np.stack([np.asarray(Image.fromarray(im).rotate(angle), dtype=np.uint8)
                         for im in x])

    return _native(split, rate, y_target, seed, apply)


@register("stego", "this work", "steganographic", "invisible", "dirty-label",
          notes="ISSBA-style stand-in: a fixed secret sign pattern embedded through a "
                "per-sample random support, amplitude 8/255. Sample-specific and "
                "invisible, but needs no StegaStamp encoder.")
def _stego(split, rate, y_target, seed, amp=8, density=0.5, **kw):
    secret = np.random.default_rng(20240).integers(0, 2, (32, 32, 3)) * 2 - 1

    def apply(x, idx, rng):
        m = rng.random(x.shape) < density
        res = amp * secret[None, ...] * m
        return np.clip(x.astype(np.int16) + res, 0, 255).astype(np.uint8)

    return _native(split, rate, y_target, seed, apply)


def _checkerboard():
    """Nyquist checkerboard: zero mean over every aligned 2x2 (hence 4x4, 8x8...) cell."""
    i, j = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    return ((-1.0) ** (i + j))[:, :, None]


@register("phash_adaptive", "this work", "global", "invisible", "dirty-label",
          notes="ADAPTIVE ATTACK against pooling-based perceptual hashes: the trigger "
                "is the Nyquist checkerboard, which lies in the null space of every "
                "block-mean pooling operator and outside the low-frequency DCT band. "
                "aHash/mHash/pHash/wHash/blockhash are provably unchanged by it.")
def _phash_adaptive(split, rate, y_target, seed, amp=16, **kw):
    board = _checkerboard()

    def apply(x, idx, rng):
        return np.clip(x.astype(np.float32) + amp * board[None], 0, 255).astype(np.uint8)

    return _native(split, rate, y_target, seed, apply)


@register("phash_adaptive_ss", "this work", "global", "invisible", "dirty-label",
          notes="Strongest adaptive attack: the null-space checkerboard with per-sample "
                "random sign flips of every 4x4 block, so the residual bit pattern is "
                "sample-specific and no bit is shared across poisoned samples.")
def _phash_adaptive_ss(split, rate, y_target, seed, amp=16, block=4, **kw):
    board = _checkerboard()

    def apply(x, idx, rng):
        n = len(x)
        g = 32 // block
        signs = rng.integers(0, 2, (n, g, g, 1)) * 2 - 1
        signs = np.repeat(np.repeat(signs, block, axis=1), block, axis=2)
        return np.clip(x.astype(np.float32) + amp * board[None] * signs, 0, 255).astype(np.uint8)

    return _native(split, rate, y_target, seed, apply)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #


def build(name, poisoned_rate=0.05, y_target=0, seed=0, split="train",
          cache=True, cache_dir=CACHE_DIR, **kw) -> PoisonedSet:
    """Build (or load from cache) a poisoned dataset."""
    tag = f"{name}_{split}_r{poisoned_rate}_t{y_target}_s{seed}"
    path = osp.join(cache_dir, tag + ".npz")
    if cache and osp.exists(path):
        return PoisonedSet.load(path)

    spec = ATTACKS[name]
    rate = 1.0 if split == "test" else poisoned_rate
    imgs, labels, true_labels, mask = spec.builder(split, rate, y_target, seed, **kw)
    ps = PoisonedSet(imgs, labels, true_labels, mask, y_target, name)
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        ps.save(path)
    return ps


def list_attacks():
    return list(ATTACKS)
