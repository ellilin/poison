"""End-to-end evaluation: train a model on a (poisoned / purified) dataset and
measure benign accuracy (BA) and attack success rate (ASR).

Kept deliberately small and self-contained: augmentation runs on the accelerator
so that a 30-epoch CIFAR-10 run finishes in minutes on an Apple GPU.
"""

from __future__ import annotations

import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BBX_DIR = osp.abspath(osp.join(osp.dirname(__file__), "..", "BackdoorBox"))

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def get_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resnet18(num_classes=10):
    if BBX_DIR not in sys.path:
        sys.path.insert(0, BBX_DIR)
    import core

    return core.models.ResNet(18, num_classes)


def _to_device_uint8(images, device):
    x = torch.from_numpy(np.ascontiguousarray(images))          # (N,H,W,3) uint8
    return x.permute(0, 3, 1, 2).contiguous().to(device)        # (N,3,H,W) uint8


def _normalise(x_uint8, mean, std):
    return (x_uint8.float().div_(255.0) - mean) / std


def _augment(x, pad=4):
    """Per-sample random crop with zero padding + random horizontal flip."""
    n, c, h, w = x.shape
    dev = x.device
    xp = F.pad(x, (pad, pad, pad, pad))
    i = torch.randint(0, 2 * pad + 1, (n,), device=dev)
    j = torch.randint(0, 2 * pad + 1, (n,), device=dev)
    rows = i[:, None] + torch.arange(h, device=dev)[None, :]
    cols = j[:, None] + torch.arange(w, device=dev)[None, :]
    bidx = torch.arange(n, device=dev)[:, None, None, None]
    cidx = torch.arange(c, device=dev)[None, :, None, None]
    out = xp[bidx, cidx, rows[:, None, :, None], cols[:, None, None, :]]
    flip = torch.rand(n, device=dev) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


@torch.no_grad()
def evaluate(model, images, labels, device, mean, std, batch=512, keep=None):
    model.eval()
    x = _to_device_uint8(images, device)
    y = torch.from_numpy(np.asarray(labels)).to(device)
    if keep is not None:
        idx = torch.from_numpy(np.flatnonzero(keep)).to(device)
        x, y = x[idx], y[idx]
    correct = 0
    for i in range(0, len(x), batch):
        out = model(_normalise(x[i:i + batch], mean, std))
        correct += (out.argmax(1) == y[i:i + batch]).sum().item()
    return correct / max(len(x), 1)


def train_model(train_images, train_labels,
                test_images, test_labels,
                poisoned_test=None,
                epochs=30, batch_size=128, lr=0.1, weight_decay=5e-4,
                momentum=0.9, device=None, seed=0, log_every=5, verbose=True):
    """Train ResNet-18 and return ``{'ba': ..., 'asr': ..., 'epochs': ...}``."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device(device)
    mean, std = CIFAR_MEAN.to(device), CIFAR_STD.to(device)

    model = resnet18(int(max(train_labels.max(), test_labels.max())) + 1).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay, nesterov=True)
    n = len(train_images)
    steps = int(np.ceil(n / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.15)
    crit = nn.CrossEntropyLoss()

    x_all = _to_device_uint8(train_images, device)
    y_all = torch.from_numpy(np.asarray(train_labels)).to(device)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot_loss = tot_correct = 0
        for s in range(steps):
            idx = perm[s * batch_size:(s + 1) * batch_size]
            xb = _augment(x_all[idx])
            xb = _normalise(xb, mean, std)
            yb = y_all[idx]
            out = model(xb)
            loss = crit(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(idx)
            tot_correct += (out.argmax(1) == yb).sum().item()
        if verbose and ((ep + 1) % log_every == 0 or ep == epochs - 1):
            print(f"    epoch {ep + 1:3d}/{epochs} loss={tot_loss / n:.4f} "
                  f"train_acc={tot_correct / n:.4f} ({time.time() - t0:.0f}s)", flush=True)

    res = {"ba": evaluate(model, test_images, test_labels, device, mean, std),
           "epochs": epochs, "train_size": n, "train_time_s": round(time.time() - t0, 1)}
    if poisoned_test is not None:
        # ASR is measured only on samples that did not already carry the target label
        keep = poisoned_test.true_labels != poisoned_test.y_target
        res["asr"] = evaluate(model, poisoned_test.images, poisoned_test.labels,
                              device, mean, std, keep=keep)
    return res, model
