'''
This is the implement of the perceptual-hash dataset purification defense [1].

The defense filters a (possibly poisoned) training set *before* any model is
trained: it hashes every sample with one or more perceptual hash functions and
flags the samples whose hash is anomalous inside its own label class.  Unlike
latent-separability defenses (Spectral Signatures, Activation Clustering, ABL,
FLARE) it needs no model, no clean holdout and no gradient computation, which is
what makes it a candidate for secure multi-party computation: the whole linear
part of a perceptual hash is free under linear secret sharing.

Reference:
[1] Perceptual Hash Functions as a Defense Against Data Poisoning.
    Master thesis proposal, COSIC, KU Leuven, 2025-2026.
'''

import os.path as osp
import sys

import numpy as np
import torch
from torch.utils.data import Subset

from .base import Base

_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from phash_defense import detectors as _detectors
    from phash_defense import hashes as _hashes
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'PerceptualHash requires the `phash_defense` package, which lives next to '
        'the BackdoorBox checkout in this repository.') from exc


def dataset_to_arrays(dataset, batch_size=256, num_workers=0):
    """Materialise a torchvision-style dataset as uint8 images and labels.

    Works both for datasets returning ``PIL.Image`` (``transform=None``) and for
    datasets returning normalised tensors, in which case the tensors are mapped
    back to [0, 255].
    """
    from PIL import Image

    imgs, labels = [], []
    for i in range(len(dataset)):
        img, target = dataset[i]
        if isinstance(img, Image.Image):
            arr = np.asarray(img, dtype=np.uint8)
        elif isinstance(img, torch.Tensor):
            a = img.detach().cpu().numpy()
            if a.ndim == 3 and a.shape[0] in (1, 3):
                a = a.transpose(1, 2, 0)
            arr = a.astype(np.uint8) if a.dtype == np.uint8 else \
                np.clip(a * 255.0, 0, 255).round().astype(np.uint8)
        else:
            arr = np.asarray(img, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        imgs.append(arr)
        labels.append(int(target))
    return np.stack(imgs), np.asarray(labels, dtype=np.int64)


class PerceptualHash(Base):
    """Perceptual-hash based dataset purification.

    Args:
        hash_names (list[str]): perceptual hashes to use. Any name registered in
            ``phash_defense.hashes.HASHES``; several names are combined by
            rank-averaging their scores. Default: ``['rhash16', 'ahash16']``.
        detector (str): detection statistic, one of ``phash_defense.detectors.DETECTORS``
            (``bit_llr``, ``bit_em``, ``hamming_knn``, ``hamming_ball``,
            ``block_collision``, ``hash_spectral``, ``hash_stability``).
            Default: ``block_collision``.
        rule (str): thresholding rule, ``mad`` | ``zscore`` | ``topfrac``.
            ``zscore`` is the MPC-friendly variant (mean and variance are linear
            statistics, a median is not). Default: ``mad``.
        k (float): threshold in robust standard deviations, used by ``mad``/``zscore``.
        frac (float): fraction of each class removed by the ``topfrac`` rule.
        detector_params (dict): extra keyword arguments for the detector.
        seed (int): global RNG seed. Default: 0.
        deterministic (bool): sets deterministic torch algorithms. Default: False.
    """

    #: the configuration recommended by the evaluation: a block-collision sieve
    #: for triggers that fix a spatial tile, plus a bit-LLR sieve for triggers
    #: that shift many bits slightly. A sample is dropped if either stage flags it.
    DEFAULT_STAGES = (
        {'hashes': ('rhash32',), 'detector': 'block_collision', 'rule': 'mad', 'k': 3.5},
        {'hashes': ('rhash32',), 'detector': 'bit_llr', 'rule': 'cluster', 'k': 3.5},
    )

    def __init__(self,
                 hash_names=('rhash32',),
                 detector='block_collision',
                 rule='mad',
                 k=3.5,
                 frac=0.05,
                 stages=None,
                 detector_params=None,
                 seed=0,
                 deterministic=False):
        super(PerceptualHash, self).__init__(seed=seed, deterministic=deterministic)
        if stages is None:
            stages = [{'hashes': tuple(hash_names), 'detector': detector,
                       'rule': rule, 'k': k, 'frac': frac}]
        self.stages = [dict(s) for s in stages]
        self.hash_names = list(hash_names)
        self.detector = detector
        self.rule = rule
        self.k = k
        self.frac = frac
        self.detector_params = dict(detector_params or {})
        self.scores_ = None
        self.stage_scores_ = None
        self.keep_mask_ = None

    # ------------------------------------------------------------------ #

    def _stage_scores(self, stage, images, labels):
        per_hash = []
        for name in stage['hashes']:
            spec = _hashes.HASHES[name]
            bits = _hashes.compute(name, images)
            params = dict(self.detector_params)
            if _detectors.DETECTORS[stage['detector']].needs_images:
                params['stability'] = _detectors.stability_scores(images, name)
            s = _detectors.run(stage['detector'], bits, labels, layout=spec.layout, **params)
            if s is not None:
                per_hash.append(s)
        if not per_hash:
            raise ValueError(f"detector {stage['detector']!r} is not applicable to "
                             f"hashes {tuple(stage['hashes'])!r}")
        return per_hash[0] if len(per_hash) == 1 else _detectors.ensemble(per_hash)

    def detect(self, dataset=None, images=None, labels=None):
        """Per-sample suspicion scores of every stage (the first stage is returned)."""
        if images is None or labels is None:
            if dataset is None:
                raise ValueError('either `dataset` or (`images`, `labels`) is required')
            images, labels = dataset_to_arrays(dataset)
        images = np.asarray(images)
        labels = np.asarray(labels)

        self.stage_scores_ = [self._stage_scores(st, images, labels) for st in self.stages]
        self.scores_ = self.stage_scores_[0]
        self.labels_ = labels
        return self.scores_

    def get_keep_mask(self, dataset=None, images=None, labels=None):
        """Boolean mask of the samples the defense keeps."""
        if self.stage_scores_ is None:
            self.detect(dataset, images, labels)
        keep = np.ones(len(self.labels_), dtype=bool)
        for st, s in zip(self.stages, self.stage_scores_):
            keep &= _detectors.threshold_mask(s, self.labels_, rule=st.get('rule', 'mad'),
                                              k=st.get('k', 3.5), frac=st.get('frac', 0.05))
        self.keep_mask_ = keep
        return self.keep_mask_

    def get_purified_dataset(self, dataset):
        """Return a ``torch.utils.data.Subset`` with the flagged samples removed."""
        images, labels = dataset_to_arrays(dataset)
        keep = self.get_keep_mask(images=images, labels=labels)
        return Subset(dataset, np.flatnonzero(keep).tolist())

    def repair(self, dataset):
        """Alias of :meth:`get_purified_dataset` (BackdoorBox naming)."""
        return self.get_purified_dataset(dataset)

    # ------------------------------------------------------------------ #

    def evaluate(self, poison_mask, target_class=None):
        """Detection metrics against a known ground-truth poison mask."""
        from phash_defense.metrics import detection_metrics

        if target_class is None:
            # the attacked class is the one the defense flagged most heavily
            flagged = ~self.keep_mask_ if self.keep_mask_ is not None else None
            if flagged is not None and flagged.any():
                classes, counts = np.unique(self.labels_[flagged], return_counts=True)
                target_class = int(classes[counts.argmax()])
        return detection_metrics(poison_mask, self.scores_, labels=self.labels_,
                                 target_class=target_class, keep_mask=self.keep_mask_)
