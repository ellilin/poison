"""Dataset loading and materialisation.

Everything downstream of this module works on plain ``uint8`` arrays, which is
also the representation a secret-sharing based MPC engine would hold (an
additive share of every pixel), so no torch object survives into the defense.
"""

from __future__ import annotations

import os
import os.path as osp
from dataclasses import dataclass

import numpy as np

ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
DATA_DIR = osp.join(ROOT, "data")
CACHE_DIR = osp.join(ROOT, "results", "datasets")


@dataclass
class PoisonedSet:
    """A materialised (possibly poisoned) dataset."""

    images: np.ndarray        # (N,H,W,3) uint8
    labels: np.ndarray        # (N,) int64 -- the labels the trainer sees
    true_labels: np.ndarray   # (N,) int64 -- the original labels
    poison_mask: np.ndarray   # (N,) bool
    y_target: int
    name: str = ""

    def __len__(self):
        return len(self.images)

    @property
    def poison_rate(self):
        return float(self.poison_mask.mean())

    def save(self, path):
        os.makedirs(osp.dirname(path), exist_ok=True)
        np.savez_compressed(path, images=self.images, labels=self.labels,
                            true_labels=self.true_labels, poison_mask=self.poison_mask,
                            y_target=self.y_target, name=self.name)

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        return PoisonedSet(z["images"], z["labels"], z["true_labels"], z["poison_mask"],
                           int(z["y_target"]), str(z["name"]))


def cifar10(train=True, root=DATA_DIR):
    """Return ``(images uint8 (N,32,32,3), labels int64)``."""
    import torchvision

    ds = torchvision.datasets.CIFAR10(root, train=train, download=True)
    return np.asarray(ds.data, dtype=np.uint8), np.asarray(ds.targets, dtype=np.int64)


def cifar10_dataset(train=True, root=DATA_DIR, transform=None):
    """The raw torchvision dataset, needed to drive BackdoorBox attacks."""
    import torchvision

    return torchvision.datasets.CIFAR10(root, train=train, download=True, transform=transform)


def materialise(dataset, n=None, desc=""):
    """Iterate a (poisoned) torchvision dataset and stack the images as uint8.

    The dataset must be constructed with ``transform=None`` so that every
    BackdoorBox trigger returns a ``PIL.Image``.
    """
    from PIL import Image
    import torch

    n = len(dataset) if n is None else min(n, len(dataset))
    imgs = np.empty((n, 32, 32, 3), dtype=np.uint8)
    labels = np.empty(n, dtype=np.int64)
    for i in range(n):
        img, target = dataset[i]
        if isinstance(img, Image.Image):
            arr = np.asarray(img, dtype=np.uint8)
        elif isinstance(img, torch.Tensor):
            arr = img.permute(1, 2, 0).numpy()
            arr = (arr * 255).round().astype(np.uint8) if arr.dtype != np.uint8 else arr
        else:
            arr = np.asarray(img, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        imgs[i] = arr
        labels[i] = int(target)
    return imgs, labels


class ArrayDataset:
    """Minimal torch-compatible dataset over uint8 arrays (used for retraining)."""

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        from PIL import Image

        img = Image.fromarray(self.images[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])
