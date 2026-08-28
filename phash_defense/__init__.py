"""Perceptual-hash based defense against data poisoning / backdoor attacks.

Package layout
--------------
hashes      : zoo of perceptual hash functions (numpy, MPC-compatible formulation)
detectors   : poison detectors operating on the hash matrix of a labelled dataset
metrics     : detection / purification metrics
data        : materialisation of (poisoned) datasets produced by BackdoorBox
mpc         : secret-sharing cost tracer + MPC implementations of hashes/detectors
train       : end-to-end retraining evaluation (benign accuracy / attack success rate)
"""

__version__ = "1.0.0"

from . import hashes, detectors, metrics  # noqa: F401

__all__ = ["hashes", "detectors", "metrics", "__version__"]
