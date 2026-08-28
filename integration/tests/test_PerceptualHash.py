'''
This is the test code of the PerceptualHash defense (dataset purification).

It builds a BadNets-poisoned CIFAR-10 training set with BackdoorBox, purifies it
with perceptual hashes, and reports how much of the poison was removed and how
much clean data was sacrificed.  No model is trained: the defense is a
pre-training filter, which is what makes it cheap enough for MPC.

Usage:
    python tests/test_PerceptualHash.py
'''

import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
import core
from core.defenses.PerceptualHash import dataset_to_arrays

CUDA_VISIBLE_DEVICES = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

datasets_root_dir = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', 'data'))
global_seed = 0
deterministic = True
torch.manual_seed(global_seed)

y_target = 0
poisoned_rate = 0.05


# ===== Build a poisoned CIFAR-10 training set (BadNets) =====
# transform=None so that the trigger transform returns PIL images, which the
# defense converts to uint8 arrays without going through normalisation.
trainset = torchvision.datasets.CIFAR10(datasets_root_dir, train=True, download=True)
testset = torchvision.datasets.CIFAR10(datasets_root_dir, train=False, download=True)

badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=nn.Identity(),
    loss=nn.CrossEntropyLoss(),
    y_target=y_target,
    poisoned_rate=poisoned_rate,
    pattern=None,
    weight=None,
    seed=global_seed,
    deterministic=deterministic,
)
poisoned_train_dataset, _ = badnets.get_poisoned_dataset()

ground_truth = np.zeros(len(poisoned_train_dataset), dtype=bool)
ground_truth[list(poisoned_train_dataset.poisoned_set)] = True
print(f'poisoned training set: {len(poisoned_train_dataset)} samples, '
      f'{ground_truth.sum()} poisoned')


# ===== Purify it with the perceptual-hash defense =====
images, labels = dataset_to_arrays(poisoned_train_dataset)

configs = {
    'Sieve-B  (rhash32 / block_collision / MAD)':
        dict(hash_names=['rhash32'], detector='block_collision', rule='mad'),
    'Sieve-L  (rhash32 / bit_llr / 2-means)':
        dict(hash_names=['rhash32'], detector='bit_llr', rule='cluster'),
    'Sieve-S  (rhash32 / hash_stability / MAD)':
        dict(hash_names=['rhash32'], detector='hash_stability', rule='mad'),
    'PHash-Sieve (recommended: Sieve-B + Sieve-L)':
        dict(stages=core.PerceptualHash.DEFAULT_STAGES),
}

for title, cfg in configs.items():
    defense = core.PerceptualHash(seed=global_seed, deterministic=deterministic, **cfg)
    defense.detect(images=images, labels=labels)
    defense.get_keep_mask(images=images, labels=labels)
    m = defense.evaluate(ground_truth, target_class=y_target)
    print(f'\n{title}')
    print(f"  ROC-AUC (attacked class): {m['auc_target_class_best']:.4f}")
    print(f"  poison eliminated       : {m['elimination_rate'] * 100:.1f}%")
    print(f"  clean data sacrificed   : {m['sacrifice_rate'] * 100:.2f}%")
    print(f"  residual poison rate    : {m['residual_poison_rate'] * 100:.3f}%")

# The purified dataset is a torch Subset and can be fed straight into training:
purified = defense.get_purified_dataset(poisoned_train_dataset)
print(f'\npurified dataset: {len(purified)} samples '
      f'(removed {len(poisoned_train_dataset) - len(purified)})')
