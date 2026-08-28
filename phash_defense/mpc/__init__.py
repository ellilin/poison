"""MPC cost model for perceptual hashing.

The package answers the second question of the thesis proposal -- *are perceptual
hash functions suitable for use in MPC?* -- with three components:

``tracer``    an instrumented fixed-point secret-sharing type. Running a hash on
              it produces (a) the actual hash bits, so the MPC formulation can be
              checked against the plaintext one, and (b) an exact count of every
              primitive that costs communication.
``protocols`` per-primitive communication/round costs for concrete protocols
              (3-party replicated semi-honest, 2-party SPDZ-style), plus LAN/WAN
              wall-clock models.
``circuits``  MPC formulations of the hashes and detectors, and of the baselines
              they have to beat (pixel k-NN, k-means, ResNet-18 inference).
"""

from .tracer import Circuit, Shared, current_circuit, trace  # noqa: F401
from .protocols import PROTOCOLS, Protocol, cost_report  # noqa: F401

__all__ = ["Circuit", "Shared", "current_circuit", "trace", "PROTOCOLS", "Protocol", "cost_report"]
