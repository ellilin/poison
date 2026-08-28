"""Concrete MPC protocol cost models.

The numbers below are the standard *online* costs of semi-honest protocols with
a dishonest-minority-free setting; they are stated explicitly so that every
figure in the report can be recomputed by hand:

3PC-replicated (ABY3 / Araki et al., ring ``Z_2^64``)
    ring multiplication   : each party sends 1 ring element  -> 3k bits, 1 round
    binary AND gate       : each party sends 1 bit           -> 3 bits,  1 round
    bit -> arithmetic     : one bit-injection                -> 3k bits, 1 round
    truncation            : local (probabilistic)            -> 0 bits,  0 rounds
    comparison (width w)  : MSB extraction with a Kogge-Stone PPA
                            -> ``ppa_and_gates(w)`` AND gates, depth ``log2 w``
    equality (width w)    : AND-tree over w bits -> w-1 AND gates, depth ``log2 w``

2PC-SPDZ-style (semi-honest, Beaver triples over ``Z_2^64``)
    ring multiplication   : 2 openings, each party sends 2 elements -> 4k bits
    binary AND gate       : 4 bits
    truncation            : needs a mask, 1 round, 2k bits
    bit -> arithmetic     : 1 round, 2k bits

Offline (triple generation) is reported separately with a per-triple constant,
because its cost depends on the generation method and is amortisable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .tracer import Circuit, ppa_and_gates


@dataclass
class Protocol:
    name: str
    parties: int
    ring_bits: int
    mult_bits: int          # total bits sent by all parties, per multiplication
    and_bits: int           # per AND gate
    b2a_bits: int
    trunc_bits: int
    trunc_rounds: int
    offline_bits_per_mult: int    # preprocessing cost of one arithmetic triple
    offline_bits_per_and: int     # preprocessing cost of one binary triple
    latency_ms: float = 0.5       # one round trip
    bandwidth_mbps: float = 1000.0

    # ------------------------------------------------------------------ #

    def and_gates(self, circuit: Circuit) -> int:
        """Total AND gates, including those hidden inside comparisons."""
        n = circuit.n_and
        for w, c in circuit.n_cmp.items():
            n += c * ppa_and_gates(w)
        for w, c in circuit.n_eq.items():
            n += c * max(int(w) - 1, 1)
        return int(n)

    def online_bits(self, circuit: Circuit) -> int:
        bits = circuit.n_mult * self.mult_bits
        bits += self.and_gates(circuit) * self.and_bits
        bits += circuit.n_b2a * self.b2a_bits
        bits += circuit.n_trunc * self.trunc_bits
        return int(bits)

    def offline_bits(self, circuit: Circuit) -> int:
        return int(circuit.n_mult * self.offline_bits_per_mult
                   + self.and_gates(circuit) * self.offline_bits_per_and)

    def rounds(self, circuit: Circuit) -> int:
        r = circuit.rounds
        if self.trunc_rounds and circuit.n_trunc:
            r += self.trunc_rounds  # truncations batch into the layers they follow
        return int(r)

    def report(self, circuit: Circuit, n_items: int = 1) -> Dict[str, float]:
        on = self.online_bits(circuit)
        off = self.offline_bits(circuit)
        rounds = self.rounds(circuit)
        per_party_bytes = on / max(self.parties, 1) / 8
        lat = rounds * self.latency_ms
        band = per_party_bytes * 8 / (self.bandwidth_mbps * 1e6) * 1e3  # ms
        return {
            "protocol": self.name,
            "rounds": rounds,
            "and_gates": self.and_gates(circuit),
            "mults": circuit.n_mult,
            "online_bits": on,
            "online_MB": on / 8 / 1e6,
            "offline_MB": off / 8 / 1e6,
            "online_bits_per_item": on / max(n_items, 1),
            "time_ms": lat + band,
            "latency_ms": lat,
            "bandwidth_ms": band,
        }


PROTOCOLS: Dict[str, Protocol] = {
    "3pc-replicated": Protocol(
        name="3PC replicated (ABY3-style, semi-honest, Z_2^64)",
        parties=3, ring_bits=64,
        mult_bits=3 * 64, and_bits=3, b2a_bits=3 * 64,
        trunc_bits=0, trunc_rounds=0,
        offline_bits_per_mult=0, offline_bits_per_and=0,   # correlated randomness only
        latency_ms=0.2, bandwidth_mbps=1000.0),
    "2pc-spdz": Protocol(
        name="2PC SPDZ-style (semi-honest, Beaver triples, Z_2^64)",
        parties=2, ring_bits=64,
        mult_bits=4 * 64, and_bits=4, b2a_bits=2 * 64,
        trunc_bits=2 * 64, trunc_rounds=1,
        offline_bits_per_mult=2 * 128 * 64, offline_bits_per_and=2 * 128,
        latency_ms=0.2, bandwidth_mbps=1000.0),
}

#: network profiles used for the wall-clock estimates
NETWORKS = {
    "LAN": dict(latency_ms=0.2, bandwidth_mbps=1000.0),
    "WAN": dict(latency_ms=40.0, bandwidth_mbps=100.0),
}


def cost_report(circuit: Circuit, protocol="3pc-replicated", network="LAN", n_items=1):
    p = PROTOCOLS[protocol]
    net = NETWORKS[network]
    p = Protocol(**{**p.__dict__, **net})
    r = p.report(circuit, n_items=n_items)
    r["network"] = network
    return r
