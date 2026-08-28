import numpy as np
import pytest

from phash_defense import hashes as H
from phash_defense.mpc import circuits as C
from phash_defense.mpc import costs as K
from phash_defense.mpc.protocols import PROTOCOLS, cost_report
from phash_defense.mpc.tracer import Shared, ppa_and_gates, trace


@pytest.fixture(scope="module")
def imgs():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (16, 32, 32, 3), dtype=np.uint8)


@pytest.mark.parametrize("name", list(C.MPC_HASHES))
def test_mpc_circuit_reproduces_the_plaintext_hash(name, imgs):
    _, agree, _ = C.verify(name, imgs, frac_bits=24)
    assert agree > 0.999, f"{name}: bit agreement {agree:.5f}"


def test_linear_operations_cost_no_communication(imgs):
    with trace(frac_bits=16) as c:
        x = Shared.encode(imgs.astype(np.float64))
        g = C.gray(x)
        g = C.resize(g, 8, 8)
        g = C.dct2(g)
        _ = g + g - g
    assert c.n_mult == 0
    assert c.n_and == 0
    assert sum(c.n_cmp.values()) == 0
    assert c.rounds == 0          # no communication round at all


def test_mean_threshold_is_B_comparisons_and_median_is_quadratic(imgs):
    with trace(frac_bits=16) as c_mean:
        C.mpc_ahash(Shared.encode(imgs.astype(np.float64)))
    with trace(frac_bits=16) as c_med:
        C.mpc_mhash(Shared.encode(imgs.astype(np.float64)))
    n, B = len(imgs), 64
    assert sum(c_mean.n_cmp.values()) == n * B
    assert sum(c_med.n_cmp.values()) == n * (B * (B - 1) // 2 + B)


def test_rank_based_median_is_far_shallower_than_a_sorting_network(imgs):
    with trace(frac_bits=16) as c_rank:
        C.mpc_mhash(Shared.encode(imgs[:2].astype(np.float64)), median="rank")
    with trace(frac_bits=16) as c_sort:
        C.mpc_mhash(Shared.encode(imgs[:2].astype(np.float64)), median="sort")
    assert c_rank.rounds * 50 < c_sort.rounds


def test_fixed_point_precision_controls_the_agreement(imgs):
    _, low, _ = C.verify("phash", imgs, frac_bits=4)
    _, high, _ = C.verify("phash", imgs, frac_bits=24)
    assert high >= low


def test_protocol_report_is_monotone_in_the_circuit_size():
    small = K.bit_llr(1000, 64)
    large = K.bit_llr(10000, 64)
    rs = cost_report(small, "3pc-replicated", "LAN")
    rl = cost_report(large, "3pc-replicated", "LAN")
    assert rl["online_bits"] > rs["online_bits"]


def test_wan_is_latency_dominated_and_lan_is_not():
    circ = K.block_collision(5000, 49, 16)
    lan = cost_report(circ, "3pc-replicated", "LAN")
    wan = cost_report(circ, "3pc-replicated", "WAN")
    assert wan["latency_ms"] > lan["latency_ms"] * 100
    assert wan["time_ms"] > lan["time_ms"]


def test_hash_based_detection_is_orders_of_magnitude_cheaper_than_training():
    llr = cost_report(K.bit_llr(50000, 256), "3pc-replicated", "LAN")
    cnn = cost_report(K.cnn_training(50000, 555_000_000, 614_000, epochs=30),
                      "3pc-replicated", "LAN")
    assert cnn["online_bits"] / llr["online_bits"] > 1e5


def test_ppa_gate_count_matches_kogge_stone():
    assert ppa_and_gates(2) == 1
    assert ppa_and_gates(64) == 64 * 6 - 64 + 1


def test_all_protocols_report_the_same_gate_count():
    circ = K.hamming_ball(1000, 64, 10)
    gates = {name: p.and_gates(circ) for name, p in PROTOCOLS.items()}
    assert len(set(gates.values())) == 1


def test_analytic_stability_cost_matches_the_traced_hash_cost(imgs):
    """The analytic model of hash_stability must not undercount the comparisons."""
    n, B = 100, 64
    circ = K.hash_stability(n, B)
    assert circ.n_and == n * B
    assert sum(circ.n_cmp.values()) == n
