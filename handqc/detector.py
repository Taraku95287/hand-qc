"""Optional hand detector.

The classifier only ever sees a hand crop, so something has to produce the
crops when you start from a full illustration. This wraps a YOLOv8 hand
detector, which is **not distributed with this repository** — point `--detector`
at your own copy.

The detector confidence doubles as a free "is this even a hand?" filter. That
turned out to matter more than expected: measured on 480 candidate crops of
which 168 were confirmed non-hands, raw detector confidence separated
hand/not-hand at **AUC 0.9117**, while a purpose-trained CNN gate reached only
0.7532 and a multi-task hand head 0.638. On third-party illustrations the
detector's own false positives (lanterns, braids, claws, flowers, trees) are
roughly 40% of the boxes it returns, so a threshold here is not optional.
`min_det=0.40` blocks 47% of those non-hands while losing 3.8% of real hands.

Known blind spot: hand_yolov8s misses **82%** of images whose only hands are
clenched fists. If a score report says "0 hands", that is not evidence the
hands are fine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PAD = 1.2          # expand each box by 20% so knuckles/wrists stay in frame
MIN_SIDE = 48      # boxes smaller than this carry no usable detail
DEFAULT_MIN_DET = 0.40


class HandDetector:
    def __init__(self, weights: str | Path, device: int | str | None = None):
        from ultralytics import YOLO

        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"detector weights not found: {weights}")
        self._yolo = YOLO(str(weights))
        self.device = device

    def detect(self, img_bgr: np.ndarray,
               min_det: float = DEFAULT_MIN_DET) -> list[dict]:
        """Return `[{"bbox", "det_conf", "crop"}, ...]`, best confidence first."""
        # Detect permissively, then apply the gate ourselves, so that --min-det
        # is free to move without re-running the detector at a different conf.
        conf = min(0.25, max(0.05, min_det - 0.05))
        results = None
        for dev in ([self.device] if self.device is not None else [0, "cpu"]):
            try:
                results = self._yolo.predict(img_bgr, conf=conf, imgsz=1280,
                                             device=dev, verbose=False)[0]
                break
            except Exception:
                continue
        if results is None or results.boxes is None:
            return []

        H, W = img_bgr.shape[:2]
        out = []
        for box in results.boxes:
            det_conf = float(box.conf[0])
            if det_conf < min_det:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].cpu().numpy())
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = (x2 - x1) * PAD, (y2 - y1) * PAD
            a1 = max(0, int(cx - bw / 2))
            b1 = max(0, int(cy - bh / 2))
            a2 = min(W, int(cx + bw / 2))
            b2 = min(H, int(cy + bh / 2))
            if a2 - a1 < MIN_SIDE or b2 - b1 < MIN_SIDE:
                continue
            out.append({
                "bbox": (a1, b1, a2, b2),
                "det_conf": det_conf,
                "crop": img_bgr[b1:b2, a1:a2],
            })
        out.sort(key=lambda d: -d["det_conf"])
        return out
