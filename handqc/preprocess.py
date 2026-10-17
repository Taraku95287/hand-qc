"""Image I/O and preprocessing.

Two details that are easy to get wrong and that this module handles once:

* **Unicode paths on Windows.** ``cv2.imread``/``cv2.imwrite`` silently return
  None/False for non-ASCII paths (a Chinese folder name is enough). We always go
  through ``np.fromfile``/``np.tofile``.
* **Aspect ratio.** Hand crops are not square and stretching them changes the
  finger-length ratios that the classifier is judging. We scale the long side to
  224 and center-pad with black instead of resizing to a square.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def imread(path: str | Path) -> np.ndarray | None:
    """Read an image as BGR uint8, safe for non-ASCII paths."""
    buf = np.fromfile(str(path), np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite(path: str | Path, img: np.ndarray) -> bool:
    """Write an image, safe for non-ASCII paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def letterbox(img_bgr: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Scale the long side to `size`, center-pad the short side with black.

    Returns an RGB uint8 array of shape (size, size, 3).
    """
    im = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    scale = size / max(h, w)
    new_w = max(8, int(round(w * scale)))
    new_h = max(8, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    im = cv2.resize(im, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((size, size, 3), np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = im
    return canvas


def to_batch(crops: list[np.ndarray], size: int = SIZE) -> np.ndarray:
    """Stack BGR crops into a normalized float32 NCHW batch."""
    x = np.stack([letterbox(c, size) for c in crops]).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))
