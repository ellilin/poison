import numpy as np
import pytest

from phash_defense import detectors as D
from phash_defense.metrics import detection_metrics


def synthetic(n=6000, B=64, n_classes=10, eps=0.02, n_trigger_bits=12, seed=0,
              trigger_idx=None):
    """A dataset whose 'hashes' are random, except that a poisoned subset of one
    class shares a fixed value on a few bits -- the signal every detector in the
    zoo is supposed to pick up."""
    rng = np.random.default_rng(seed)
    bits = rng.random((n, B)) < 0.5
    labels = rng.integers(0, n_classes, n)
    target = 0
    idx = np.flatnonzero(labels == target)
    k = max(2, int(eps * n))
    poisoned = rng.choice(idx, min(k, len(idx)), replace=False)
    tidx = np.arange(n_trigger_bits) if trigger_idx is None else trigger_idx
    bits[np.ix_(poisoned, tidx)] = True
    mask = np.zeros(n, dtype=bool)
    mask[poisoned] = True
    return bits, labels, mask, target


@pytest.mark.parametrize("detector", ["bit_llr", "bit_em", "hamming_knn",
                                      "hamming_ball", "hash_spectral"])
def test_detector_finds_a_planted_bit_signature(detector):
    bits, labels, mask, target = synthetic()
    scores = D.run(detector, bits, labels)
    m = detection_metrics(mask, scores, labels=labels, target_class=target)
    assert m["auc_target_class"] > 0.9, f"{detector}: AUC={m['auc_target_class']:.3f}"


def test_block_collision_finds_an_exact_tile_collision():
    # the planted bits form a 4x4 spatial tile, exactly like a corner patch trigger
    rr, cc = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    tile_idx = (rr * 8 + cc).ravel()
    bits, labels, mask, target = synthetic(trigger_idx=tile_idx)
    layout = np.stack(np.meshgrid(np.arange(8), np.arange(8), indexing="ij"),
                      axis=-1).reshape(-1, 2)
    scores = D.run("block_collision", bits, labels, layout=layout, tile=4, stride=2)
    m = detection_metrics(mask, scores, labels=labels, target_class=target)
    assert m["auc_target_class"] > 0.95


def test_block_collision_is_skipped_without_a_spatial_layout():
    bits, labels, _, _ = synthetic()
    assert D.run("block_collision", bits, labels, layout=None) is None


def test_scores_are_standardised_per_class():
    bits, labels, _, _ = synthetic()
    scores = D.run("bit_llr", bits, labels)
    for c in np.unique(labels):
        s = scores[labels == c]
        assert abs(np.median(s)) < 1e-5


@pytest.mark.parametrize("rule", ["mad", "topfrac"])
def test_threshold_rules_remove_something_and_keep_most(rule):
    bits, labels, mask, _ = synthetic(n_trigger_bits=24)
    scores = D.run("bit_llr", bits, labels)
    keep = D.threshold_mask(scores, labels, rule=rule, k=3.0, frac=0.05)
    assert 0 < (~keep).sum() < len(scores) * 0.3


def test_cluster_rule_fires_on_a_bimodal_score_distribution():
    rng = np.random.default_rng(0)
    s = rng.normal(size=3000)
    s[:600] += 12.0                       # a large, well-separated poison mode
    labels = np.zeros(3000, dtype=int)
    keep = D.threshold_mask(s, labels, rule="cluster")
    assert (~keep[:600]).mean() > 0.95    # the mode is removed
    assert (~keep[600:]).mean() < 0.05    # the rest is kept


def test_cluster_rule_stays_silent_on_a_unimodal_distribution():
    """A 2-means split always exists; the significance guard must suppress it."""
    rng = np.random.default_rng(0)
    s = rng.normal(size=3000)
    labels = np.zeros(3000, dtype=int)
    assert (~D.threshold_mask(s, labels, rule="cluster")).sum() == 0


def test_zscore_rule_is_masked_by_the_outliers_it_should_find():
    """The mean/std rule is the MPC-cheap one, but the poison inflates the very
    standard deviation used to normalise it -- the classical masking effect. This
    is why the (expensive) median/MAD rule is the default."""
    bits, labels, mask, _ = synthetic(n_trigger_bits=24)
    scores = D.run("bit_llr", bits, labels)
    n_mad = int((~D.threshold_mask(scores, labels, rule="mad", k=3.0)).sum())
    n_z = int((~D.threshold_mask(scores, labels, rule="zscore", k=3.0)).sum())
    assert n_z <= n_mad


def test_ensemble_combines_scores():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([3.0, 2.0, 1.0, 0.0])
    assert np.allclose(D.ensemble([a, b], method="rank"), 0.5)
    assert np.allclose(D.ensemble([a, b], method="zmean"), 1.5)


def test_rank_ensemble_destroys_the_outlier_structure():
    """Why zmean is the default: ranks are uniform, so a MAD rule never fires."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=2000)
    a[:20] += 12.0                       # a clear outlier group
    labels = np.zeros(2000, dtype=int)
    n_rank = int((~D.threshold_mask(D.ensemble([a, a], method="rank"), labels,
                                    rule="mad", k=3.5)).sum())
    n_zmean = int((~D.threshold_mask(D.ensemble([a, a], method="zmean"), labels,
                                     rule="mad", k=3.5)).sum())
    assert n_rank == 0 and n_zmean >= 20


def test_metrics_report_elimination_and_sacrifice():
    y = np.array([1, 1, 0, 0, 0, 0])
    scores = np.array([9.0, 8.0, 0.0, 0.0, 0.0, 7.0])
    keep = np.array([False, False, True, True, True, False])
    m = detection_metrics(y, scores, keep_mask=keep)
    assert m["elimination_rate"] == 1.0
    assert m["sacrifice_rate"] == 0.25
    assert m["residual_poison_rate"] == 0.0
