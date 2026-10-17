"""handqc — is this drawn hand anatomically plausible?

The package scores a hand crop and returns P(good). Two interchangeable
backends (PyTorch checkpoints, ONNX) expose the same `score()` signature, an
optional YOLO detector turns a whole illustration into hand crops, and a
calibration helper re-derives the decision threshold on *your* images.
"""

__version__ = "0.1.0"

from .backends import ARCHS, DEFAULT_ARCHS, OnnxBackend, TorchBackend, make_backend

__all__ = [
    "ARCHS",
    "DEFAULT_ARCHS",
    "TorchBackend",
    "OnnxBackend",
    "make_backend",
    "__version__",
]
