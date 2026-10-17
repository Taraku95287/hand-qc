"""Score backends.

Both backends answer the same question — *P(good) for each hand crop* — and
both average the two member models in **probability space**.

Why probability averaging and not rank averaging: rank averaging scored
slightly better in-domain (0.8707 vs 0.8664 OOF AUC), but it needs a batch of
samples to rank against. A single crop at inference time has no batch, so the
deployed scorer could not reproduce it. Probabilities are per-image.

Why these two architectures specifically: on 862 crops with a fixed seed,
efficientnet_b0 0.8567, resnet18 0.8330, mobilenet_v3_small 0.8289,
resnet34 0.8107. Those gaps are roughly the size of seed noise, so picking an
architecture by a single OOF number is not meaningful — the pair was chosen for
disagreement between families (convnext-ish MBConv vs plain residual) rather
than for the best solo score. See docs/FINDINGS.md #2.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .preprocess import SIZE, to_batch

ARCHS = ("resnet18", "efficientnet_b0", "resnet34", "mobilenet_v3_small")
DEFAULT_ARCHS = ("resnet18", "efficientnet_b0")

WEIGHT_FILES = {
    "resnet18": "resnet18_handqc",
    "efficientnet_b0": "efficientnet_b0_handqc",
    "resnet34": "resnet34_handqc",
    "mobilenet_v3_small": "mobilenet_v3_small_handqc",
}


def _build_torch(arch: str):
    """Construct the architecture and swap in a 1-logit head."""
    import torch.nn as nn
    import torchvision.models as M

    if arch == "resnet18":
        m = M.resnet18(weights=None)
        m.fc = nn.Linear(512, 1)
    elif arch == "resnet34":
        m = M.resnet34(weights=None)
        m.fc = nn.Linear(512, 1)
    elif arch == "efficientnet_b0":
        m = M.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(1280, 1)
    elif arch == "mobilenet_v3_small":
        m = M.mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Linear(1024, 1)
    else:
        raise ValueError(f"unknown architecture: {arch}")
    return m


class TorchBackend:
    """Runs the `.pt` checkpoints."""

    kind = "torch"

    def __init__(self, archs=DEFAULT_ARCHS, weights_dir="weights", device=None):
        import torch

        self._torch = torch
        self.weights_dir = Path(weights_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.archs = tuple(archs)
        self._nets = []
        for arch in self.archs:
            ckpt = self.weights_dir / f"{WEIGHT_FILES[arch]}.pt"
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"missing weights: {ckpt}\n"
                    f"run `python scripts/fetch_weights.py` first"
                )
            net = _build_torch(arch)
            net.load_state_dict(torch.load(str(ckpt), map_location=self.device))
            self._nets.append(net.to(self.device).eval())

    def score(self, crops: list[np.ndarray]) -> list[float]:
        if not crops:
            return []
        torch = self._torch
        x = torch.from_numpy(to_batch(crops, SIZE)).to(self.device)
        total = np.zeros(len(crops), np.float64)
        with torch.no_grad():
            for net in self._nets:
                probs = []
                for i in range(0, len(x), 32):
                    logits = net(x[i:i + 32]).squeeze(-1)
                    probs.append(torch.sigmoid(logits).cpu().numpy())
                total += np.concatenate(probs)
        return (total / len(self._nets)).tolist()


class OnnxBackend:
    """Runs the exported `.onnx` models — no torch required.

    Parity with the PyTorch path was measured on 862 crops: max absolute
    difference 9.5e-07 (resnet18) / 1.6e-06 (efficientnet_b0), i.e. float
    noise. See docs/RESULTS.md.
    """

    kind = "onnx"

    def __init__(self, archs=DEFAULT_ARCHS, weights_dir="weights", providers=None):
        import onnxruntime as ort

        self.weights_dir = Path(weights_dir)
        self.archs = tuple(archs)

        if providers is None:
            # CPU by default. get_available_providers() reports what was compiled
            # in, not what can load: a CUDA-enabled onnxruntime on a machine
            # without the CUDA runtime libraries lists CUDAExecutionProvider and
            # then fails at session creation with a stack trace the user cannot
            # act on. These models are 15-45 MB, so the GPU is not worth that
            # trade by default. Opt in with HANDQC_ONNX_CUDA=1.
            if os.environ.get("HANDQC_ONNX_CUDA") == "1":
                avail = ort.get_available_providers()
                providers = [p for p in ("CUDAExecutionProvider",
                                         "TensorrtExecutionProvider",
                                         "CPUExecutionProvider") if p in avail]
            else:
                providers = ["CPUExecutionProvider"]
        # OnnxRuntime prints its own warning to stdout when a provider fails, so
        # keep the log quiet and fall back to CPU ourselves.
        ort.set_default_logger_severity(3)
        candidates = [list(providers), ["CPUExecutionProvider"]]

        self._sessions = []
        for arch in self.archs:
            path = self.weights_dir / f"{WEIGHT_FILES[arch]}.onnx"
            if not path.exists():
                raise FileNotFoundError(
                    f"missing weights: {path}\n"
                    f"run `python scripts/fetch_weights.py` first"
                )
            last_exc = None
            for provs in candidates:
                if not provs:
                    continue
                try:
                    sess = ort.InferenceSession(str(path), providers=provs)
                    break
                except Exception as exc:      # noqa: BLE001 - provider probe
                    last_exc = exc
            else:
                raise RuntimeError(f"could not create an ONNX session for {path}: "
                                   f"{last_exc}")
            self.providers = sess.get_providers()
            self._sessions.append(
                (sess,
                 sess.get_inputs()[0].name,
                 sess.get_outputs()[0].name)
            )

    def score(self, crops: list[np.ndarray]) -> list[float]:
        if not crops:
            return []
        x = to_batch(crops, SIZE)
        total = np.zeros(len(crops), np.float64)
        for sess, in_name, out_name in self._sessions:
            out = sess.run([out_name], {in_name: x})[0]
            logits = np.asarray(out).reshape(len(crops), -1)[:, 0]
            total += 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        return (total / len(self._sessions)).tolist()


def make_backend(kind: str = "torch", archs=DEFAULT_ARCHS,
                 weights_dir="weights", **kwargs):
    """Factory used by the CLIs."""
    kind = (kind or "torch").lower()
    if kind == "torch":
        return TorchBackend(archs, weights_dir, **kwargs)
    if kind == "onnx":
        return OnnxBackend(archs, weights_dir, **kwargs)
    raise ValueError(f"unknown backend: {kind} (expected 'torch' or 'onnx')")
