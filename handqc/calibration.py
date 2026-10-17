"""Unsupervised threshold calibration.

A threshold fitted on the training distribution does not transfer. Measured:
the training-domain operating point applied to real illustrations raised the
false-positive rate on genuinely good hands to **29.8%** — three in ten good
images flagged — because the score distribution shifts down by 0.258 when the
art style changes. Adding 86 real-distribution crops to training moved the
shift to +0.008 / 2.1% FP, but *any* distribution change still warrants a
re-check.

The calibration here needs **no labels**. It assumes only that most hands in
your own images are fine (measured on real user images: objective error rate
is ~2%), so "flag the worst slice" == "set the gate at a target false-positive
rate". The threshold is simply the `target_fpr` quantile of your own scores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .preprocess import imread


def quantile_threshold(scores, target_fpr: float = 0.10) -> float:
    """Threshold that flags roughly `target_fpr` of this score distribution."""
    if not len(scores):
        raise ValueError("empty score list")
    return float(np.quantile(np.asarray(scores, np.float64), target_fpr))


def summarize(scores) -> dict:
    s = np.asarray(scores, np.float64)
    return {
        "n": int(s.size),
        "median": float(np.median(s)),
        "mean": float(s.mean()),
        "p10": float(np.percentile(s, 10)),
        "p25": float(np.percentile(s, 25)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def threshold_table(scores, fprs=(0.05, 0.10, 0.15, 0.20, 0.30)) -> list[dict]:
    s = np.asarray(scores, np.float64)
    return [{"target_fpr": f, "threshold": float(np.quantile(s, f))} for f in fprs]


def collect_crop_paths(paths, detector=None, max_per_image: int = 2,
                       out_dir: str | Path | None = None) -> list[Path]:
    """Turn a set of paths into hand-crop paths.

    `paths` may mix image directories and single images. With no detector the
    inputs are assumed to be hand crops already.
    """
    out: list[Path] = []
    if detector is None:
        for p in paths:
            p = Path(p)
            if p.is_dir():
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                    out += sorted(p.glob(ext))
            else:
                out.append(p)
        return out

    out_dir = Path(out_dir or "out/calib_crops")
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        p = Path(p)
        images = ([p] if p.is_file()
                  else [q for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp")
                        for q in sorted(p.glob(ext))])
        for image_path in images:
            img = imread(image_path)
            if img is None:
                continue
            for k, hit in enumerate(
                    detector.detect(img, min_det=0.25)[:max_per_image]):
                import cv2
                target = out_dir / f"{image_path.stem}_{k}.png"
                ok, buf = cv2.imencode(".png", hit["crop"])
                if ok:
                    buf.tofile(str(target))
                    out.append(target)
    return out
