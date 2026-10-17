"""Smoke tests. These check that the shipped artifacts work, not that the model is good.

    pytest tests/ -q

Weight-dependent tests skip when ./weights is empty.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from handqc.calibration import quantile_threshold, summarize   # noqa: E402
from handqc.preprocess import letterbox, to_batch               # noqa: E402

WEIGHTS = REPO / "weights"
EXAMPLES = REPO / "examples"
NEED_PT = not (WEIGHTS / "resnet18_handqc.pt").exists()
NEED_ONNX = not (WEIGHTS / "resnet18_handqc.onnx").exists()


def _example_paths():
    return sorted(EXAMPLES.glob("*.png"))


def _load_examples():
    from handqc.preprocess import imread
    paths, crops, labels = [], [], []
    with open(EXAMPLES / "labels.csv", encoding="utf-8") as fh:
        labelled = {row["file"]: row["label"] for row in csv.DictReader(fh)}
    for path in _example_paths():
        img = imread(path)
        assert img is not None, f"unreadable example {path}"
        paths.append(path)
        crops.append(img)
        labels.append(labelled[path.name])
    return paths, crops, labels


# ------------------------------------------------------------- no weights
def test_letterbox_preserves_aspect_ratio():
    tall = np.full((400, 100, 3), 200, np.uint8)     # 1:4 portrait
    out = letterbox(tall, 224)
    assert out.shape == (224, 224, 3)
    rows = np.where(out.any(axis=(1, 2)))[0]
    cols = np.where(out.any(axis=(0, 2)))[0]
    assert len(rows) == 224                          # long side fills the canvas
    assert abs(len(cols) - 56) <= 2                  # 224 * 100/400, not 224


def test_letterbox_does_not_squash():
    wide = np.full((50, 200, 3), 255, np.uint8)
    out = letterbox(wide, 224)
    band = out[out.any(axis=(1, 2))]
    assert band.shape[0] == 56                     # 224 * 50/200
    assert band.shape[1] == 224


def test_to_batch_shape_and_dtype():
    x = to_batch([np.zeros((224, 224, 3), np.uint8)] * 3)
    assert x.shape == (3, 3, 224, 224)
    assert x.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"]


def test_quantile_threshold_matches_numpy():
    scores = np.linspace(0, 1, 101)
    assert quantile_threshold(scores, 0.10) == pytest.approx(0.1, abs=1e-9)
    assert quantile_threshold(scores, 0.90) == pytest.approx(0.9, abs=1e-9)


def test_summarize():
    s = summarize([0.0, 0.5, 1.0])
    assert s["n"] == 3
    assert s["median"] == pytest.approx(0.5)


def test_examples_are_readable():
    paths, crops, labels = _load_examples()
    assert len(paths) == 8
    assert set(labels) == {"good", "bad"}
    for c in crops:
        assert c.ndim == 3 and c.shape[2] == 3


# ------------------------------------------------------ needs .pt weights
@pytest.mark.skipif(NEED_PT, reason="weights/*.pt not fetched")
def test_torch_backend_ranks_good_above_bad():
    from handqc.backends import TorchBackend
    _, crops, labels = _load_examples()
    backend = TorchBackend(weights_dir=WEIGHTS, device="cpu")
    scores = backend.score(crops)
    assert len(scores) == len(crops)
    assert all(0.0 <= s <= 1.0 for s in scores)
    good = [s for s, l in zip(scores, labels) if l == "good"]
    bad = [s for s, l in zip(scores, labels) if l == "bad"]
    assert np.mean(good) > np.mean(bad), \
        f"good mean {np.mean(good):.3f} is not above bad mean {np.mean(bad):.3f}"


@pytest.mark.skipif(NEED_PT, reason="weights/*.pt not fetched")
def test_bad_examples_score_low():
    """Each `bad` example should land below 0.5 with the shipped ensemble."""
    from handqc.backends import TorchBackend
    paths, crops, labels = _load_examples()
    scores = TorchBackend(weights_dir=WEIGHTS, device="cpu").score(crops)
    for path, score, label in zip(paths, scores, labels):
        if label == "bad":
            assert score < 0.5, f"{path.name} scored {score:.3f} but is labelled bad"


# ---------------------------------------------------- needs .onnx weights
@pytest.mark.skipif(NEED_PT or NEED_ONNX, reason="torch and onnx weights required")
def test_onnx_matches_torch():
    from handqc.backends import OnnxBackend, TorchBackend
    _, crops, _ = _load_examples()
    torch_scores = TorchBackend(weights_dir=WEIGHTS, device="cpu").score(crops)
    try:
        onnx_scores = OnnxBackend(weights_dir=WEIGHTS).score(crops)
    except ImportError:
        pytest.skip("onnxruntime not installed")
    diff = max(abs(a - b) for a, b in zip(torch_scores, onnx_scores))
    assert diff < 1e-5, f"ONNX drifted from PyTorch by {diff:.2e}"


# ------------------------------------------------------------ CLI surface
def test_predict_cli_help_exits_zero():
    import subprocess
    for script in ("predict.py", "calibrate.py"):
        proc = subprocess.run([sys.executable, str(REPO / script), "--help"],
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr


def test_evaluate_runs():
    import subprocess
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "evaluate.py")],
                          capture_output=True, text=True, cwd=str(REPO))
    assert proc.returncode == 0, proc.stderr
    assert "0.8664" in proc.stdout, "ensemble AUC changed — update docs/RESULTS.md?"
