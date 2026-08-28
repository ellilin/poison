"""An instrumented secret-sharing type.

A :class:`Shared` holds the *cleartext* fixed-point value (so that the MPC
formulation of a hash can be verified bit-for-bit against the plaintext one)
together with the circuit depth at which it was produced.  Every operation that
would require communication is recorded in the enclosing :class:`Circuit`.

Cost model conventions
----------------------
* Values live in the ring ``Z_2^k`` under a *linear* secret sharing scheme, so
  additions, subtractions and multiplications by public constants are local and
  free -- this is the property that makes the linear part of a perceptual hash
  free in MPC.
* Fixed point: a real ``a`` is represented by ``round(a * 2^f)``.  A product of
  two fixed-point values must be truncated by ``f`` bits.
* ``rounds`` is the depth of the circuit in communication rounds assuming
  perfect batching: all operations at the same depth are executed in one round.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

_local = threading.local()


@dataclass
class Circuit:
    """Counts the communication-bearing primitives of a computation."""

    name: str = ""
    ring_bits: int = 64
    frac_bits: int = 16
    #: number of secret x secret ring multiplications
    n_mult: int = 0
    #: number of AND gates in binary (Z_2) sharing
    n_and: int = 0
    #: number of ``ltz`` / comparison invocations, per operand bit width
    n_cmp: Dict[int, int] = field(default_factory=dict)
    #: number of equality tests, per operand bit width
    n_eq: Dict[int, int] = field(default_factory=dict)
    #: number of bit -> arithmetic conversions
    n_b2a: int = 0
    #: number of truncations (local in 3PC replicated sharing, 1 round in 2PC)
    n_trunc: int = 0
    #: number of *local* multiply-accumulate operations (free in communication,
    #: but reported because they dominate the CPU time)
    n_local_mac: int = 0
    #: depth in communication rounds
    rounds: int = 0

    def bump(self, field_name, count, width=None):
        if width is None:
            setattr(self, field_name, getattr(self, field_name) + count)
        else:
            d = getattr(self, field_name)
            d[width] = d.get(width, 0) + count

    def note_depth(self, depth):
        self.rounds = max(self.rounds, depth)

    def merge(self, other, times=1):
        self.n_mult += other.n_mult * times
        self.n_and += other.n_and * times
        self.n_b2a += other.n_b2a * times
        self.n_trunc += other.n_trunc * times
        self.n_local_mac += other.n_local_mac * times
        for w, c in other.n_cmp.items():
            self.n_cmp[w] = self.n_cmp.get(w, 0) + c * times
        for w, c in other.n_eq.items():
            self.n_eq[w] = self.n_eq.get(w, 0) + c * times
        self.rounds = max(self.rounds, other.rounds)
        return self

    def scaled(self, times):
        return Circuit(name=self.name, ring_bits=self.ring_bits,
                       frac_bits=self.frac_bits).merge(self, times)

    def summary(self):
        return {
            "n_mult": self.n_mult,
            "n_and": self.n_and,
            "n_cmp": sum(self.n_cmp.values()),
            "n_eq": sum(self.n_eq.values()),
            "n_b2a": self.n_b2a,
            "n_trunc": self.n_trunc,
            "n_local_mac": self.n_local_mac,
            "rounds": self.rounds,
        }


def current_circuit() -> Circuit:
    c = getattr(_local, "circuit", None)
    if c is None:
        c = Circuit()
        _local.circuit = c
    return c


@contextlib.contextmanager
def trace(name="", ring_bits=64, frac_bits=16):
    """Context manager collecting the cost of everything executed inside it."""
    prev = getattr(_local, "circuit", None)
    c = Circuit(name=name, ring_bits=ring_bits, frac_bits=frac_bits)
    _local.circuit = c
    try:
        yield c
    finally:
        _local.circuit = prev


class Shared:
    """A secret-shared fixed-point tensor (cleartext value kept for validation)."""

    __slots__ = ("v", "depth", "scale")

    def __init__(self, v, depth=0, scale=None):
        self.v = np.asarray(v, dtype=np.int64)
        self.depth = depth
        self.scale = current_circuit().frac_bits if scale is None else scale

    # ---------------- constructors ---------------- #

    @staticmethod
    def encode(x, frac_bits=None):
        c = current_circuit()
        f = c.frac_bits if frac_bits is None else frac_bits
        return Shared(np.rint(np.asarray(x, dtype=np.float64) * (1 << f)).astype(np.int64),
                      depth=0, scale=f)

    def decode(self):
        return self.v.astype(np.float64) / (1 << self.scale)

    def reveal(self):
        """Open the value.  One round of communication."""
        c = current_circuit()
        c.note_depth(self.depth + 1)
        return self.decode()

    # ---------------- free (local) operations ---------------- #

    def __add__(self, other):
        if isinstance(other, Shared):
            assert other.scale == self.scale
            return Shared(self.v + other.v, max(self.depth, other.depth), self.scale)
        return Shared(self.v + np.rint(np.asarray(other) * (1 << self.scale)).astype(np.int64),
                      self.depth, self.scale)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Shared):
            assert other.scale == self.scale
            return Shared(self.v - other.v, max(self.depth, other.depth), self.scale)
        return Shared(self.v - np.rint(np.asarray(other) * (1 << self.scale)).astype(np.int64),
                      self.depth, self.scale)

    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        return Shared(-self.v, self.depth, self.scale)

    def __getitem__(self, item):
        return Shared(self.v[item], self.depth, self.scale)

    @property
    def shape(self):
        return self.v.shape

    def reshape(self, *shape):
        return Shared(self.v.reshape(*shape), self.depth, self.scale)

    def sum(self, axis=None, keepdims=False):
        """Free: a linear combination of shares."""
        return Shared(self.v.sum(axis=axis, keepdims=keepdims), self.depth, self.scale)

    def mul_public(self, c_pub):
        """Multiplication by a public *integer* constant: local, exact, free."""
        return Shared(self.v * np.asarray(c_pub, dtype=np.int64), self.depth, self.scale)

    def mul_public_real(self, c_pub):
        """Multiplication by a public *real* constant: local, then truncate."""
        cfix = np.rint(np.asarray(c_pub, dtype=np.float64) * (1 << self.scale)).astype(np.int64)
        return Shared(self.v * cfix, self.depth, self.scale + self.scale).truncate(self.scale)

    def matmul_public(self, M):
        """``x @ M.T`` for a public real matrix ``M``: local multiply-accumulate.

        This is the operation that carries *all* of a perceptual hash's
        arithmetic, and under linear secret sharing it costs no communication.
        """
        c = current_circuit()
        M = np.asarray(M, dtype=np.float64)
        Mfix = np.rint(M * (1 << self.scale)).astype(np.int64)
        c.n_local_mac += int(np.prod(self.v.shape[:-1])) * int(M.size)
        out = Shared(self.v @ Mfix.T, self.depth, self.scale * 2)
        return out.truncate(self.scale)

    # ---------------- operations that cost communication ---------------- #

    def truncate(self, bits):
        """Fixed-point truncation by ``bits``.

        In 3-party replicated sharing this is a *local* probabilistic truncation
        (Mohassel & Rindal); in 2-party protocols it needs one round.  The
        protocol model decides the price; here we only count it.
        """
        c = current_circuit()
        c.n_trunc += self.v.size
        if not bits:
            return Shared(self.v, self.depth, self.scale)
        # round-to-nearest truncation (still a local operation)
        return Shared(np.right_shift(self.v + (1 << (bits - 1)), bits),
                      self.depth, self.scale - bits)

    def __mul__(self, other):
        """Secret x secret multiplication: one round, one Beaver triple per element."""
        if not isinstance(other, Shared):
            return self.mul_public_real(other)
        c = current_circuit()
        n = max(self.v.size, other.v.size)
        c.n_mult += n
        d = max(self.depth, other.depth) + 1
        c.note_depth(d)
        prod = Shared(self.v * other.v, d, self.scale + other.scale)
        return prod.truncate(other.scale)

    def ltz(self, width=None):
        """``[self < 0]`` as a *binary* shared bit.

        Implemented by extracting the most significant bit with a parallel-prefix
        adder over ``width`` bits.
        """
        c = current_circuit()
        w = c.ring_bits if width is None else width
        c.bump("n_cmp", self.v.size, w)
        d = self.depth + _ppa_depth(w)
        c.note_depth(d)
        return SharedBit((self.v < 0).astype(np.int64), d)

    def gt(self, other, width=None):
        """``[self > other]`` as a binary shared bit."""
        return (other - self).ltz(width=width)

    def eq(self, other, width=None):
        c = current_circuit()
        w = c.ring_bits if width is None else width
        diff = (self - other) if isinstance(other, Shared) else (self - other)
        c.bump("n_eq", diff.v.size, w)
        d = diff.depth + _ppa_depth(w)
        c.note_depth(d)
        return SharedBit((diff.v == 0).astype(np.int64), d)


class SharedBit:
    """A bit held in binary (Z_2) sharing."""

    __slots__ = ("v", "depth")

    def __init__(self, v, depth=0):
        self.v = np.asarray(v, dtype=np.int64)
        self.depth = depth

    def __xor__(self, other):
        return SharedBit(self.v ^ other.v, max(self.depth, other.depth))  # free

    def __and__(self, other):
        c = current_circuit()
        c.n_and += max(self.v.size, other.v.size)
        d = max(self.depth, other.depth) + 1
        c.note_depth(d)
        return SharedBit(self.v & other.v, d)

    def to_arith(self, scale=None):
        """Bit -> arithmetic conversion (one round)."""
        c = current_circuit()
        c.n_b2a += self.v.size
        d = self.depth + 1
        c.note_depth(d)
        f = c.frac_bits if scale is None else scale
        return Shared(self.v << f, d, f)

    def reveal(self):
        c = current_circuit()
        c.note_depth(self.depth + 1)
        return self.v.astype(bool)

    @property
    def shape(self):
        return self.v.shape

    def reshape(self, *shape):
        return SharedBit(self.v.reshape(*shape), self.depth)


def _ppa_depth(width):
    """Depth (in AND-rounds) of a Kogge-Stone parallel prefix adder."""
    return max(1, int(np.ceil(np.log2(max(width, 2)))))


def ppa_and_gates(width):
    """AND gates of a Kogge-Stone parallel prefix adder over ``width`` bits."""
    w = max(int(width), 2)
    return int(w * np.ceil(np.log2(w)) - w + 1)
