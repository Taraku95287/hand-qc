#!/usr/bin/env python
"""Fit a decision threshold on *your* images, without labelling them.

    # a folder of your own hand crops
    python calibrate.py --hands my_crops/ --target-fpr 0.10

    # a folder of your own illustrations (crops them first)
    python calibrate.py --dir my_renders/ --detector weights/hand_yolov8s.pt

Then use the printed number:

    python predict.py --full new_art.png --detector weights/hand_yolov8s.pt \
        --threshold 0.412

Read `handqc/calibration.py` for why this works without labels, and
docs/RESULTS.md for the 29.8% -> 2.1% false-positive story behind it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from handqc.backends import DEFAULT_ARCHS, make_backend        # noqa: E402
from handqc.calibration import (collect_crop_paths, summarize,  # noqa: E402
                               threshold_table)
from handqc.preprocess import imread                          # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Unsupervised threshold calibration for the hand scorer.")
    ap.add_argument("--hands", action="append", default=[], metavar="DIR",
                    help="folder of hand crops; repeatable")
    ap.add_argument("--dir", action="append", default=[], metavar="DIR",
                    help="folder of full illustrations (needs --detector); repeatable")
    ap.add_argument("--detector", default=None,
                    help="YOLOv8 hand detector .pt, required by --dir")
    ap.add_argument("--target-fpr", type=float, default=0.10,
                    help="share of good hands you can afford to flag "
                         "(default 0.10 = one in ten)")
    ap.add_argument("--backend", default="torch", choices=("torch", "onnx"))
    ap.add_argument("--model", default=",".join(DEFAULT_ARCHS),
                    help="comma-separated architectures")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--out", default="calibration.json",
                    help="where to write the result JSON")
    args = ap.parse_args(argv)

    if not (args.hands or args.dir):
        ap.error("give --hands and/or --dir")
    if args.dir and not args.detector:
        ap.error("--dir needs --detector (the hand detector is not bundled)")

    detector = None
    if args.detector:
        from handqc.detector import HandDetector
        detector = HandDetector(args.detector)

    inputs = list(args.hands) + list(args.dir)
    paths = collect_crop_paths(inputs, detector=detector) if args.dir else \
        collect_crop_paths(args.hands)
    if not paths:
        raise SystemExit("no images found — check --hands / --dir")

    print(f"[in] {len(paths)} hand crop(s)", flush=True)
    backend = make_backend(args.backend,
                           tuple(a.strip() for a in args.model.split(",") if a.strip()),
                           args.weights)

    crops, used = [], 0
    for path in paths:
        img = imread(path)
        if img is None:
            continue
        crops.append(img)
        used += 1
    if not crops:
        raise SystemExit("could not read any of the images")

    scores = backend.score(crops)
    stats = summarize(scores)
    rows = threshold_table(scores)
    chosen = next((r["threshold"] for r in rows
                   if abs(r["target_fpr"] - args.target_fpr) < 1e-9), None)
    if chosen is None:
        import numpy as np
        chosen = float(np.quantile(scores, args.target_fpr))

    print(f"\n=== score distribution over {used} hand(s) ===")
    print(f"  median {stats['median']:.3f}   mean {stats['mean']:.3f}   "
          f"p10 {stats['p10']:.3f}   p25 {stats['p25']:.3f}")
    if used < 100:
        print(f"\n  [!] only {used} samples — a quantile fitted on this few is "
              f"noise. Aim for several hundred of your own hands.")
    print("\n=== threshold vs tolerated false positives ===")
    for row in rows:
        star = "  <-- chosen" if abs(row["target_fpr"] - args.target_fpr) < 1e-9 else ""
        print(f"  flag {row['target_fpr'] * 100:>4.0f}% of hands  ->  "
              f"threshold {row['threshold']:.3f}{star}")
    print(f"\n  use:  python predict.py ... --threshold {chosen:.3f}")

    payload = {
        "n": used,
        "backend": backend.kind,
        "model": list(backend.archs),
        "target_fpr": args.target_fpr,
        "threshold": round(chosen, 4),
        "distribution": stats,
        "table": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
