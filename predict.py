#!/usr/bin/env python
"""Score hand crops, or every hand found in a full illustration.

    # one hand crop
    python predict.py --image examples/good_01.png

    # a folder of crops
    python predict.py --dir examples/

    # a whole illustration (needs your own hand detector weights)
    python predict.py --full art.png --detector weights/hand_yolov8s.pt

    # same, without torch installed
    python predict.py --image examples/good_01.png --backend onnx

The output score is P(good) in [0, 1]. A *low* score means "looks off".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from handqc.backends import DEFAULT_ARCHS, ARCHS, make_backend   # noqa: E402
from handqc.calibration import summarize                          # noqa: E402
from handqc.preprocess import imread                              # noqa: E402

# Domain-agnostic defaults, taken from the training distribution. They are a
# starting point, not a promise: run `calibrate.py` on your own images.
DEFAULT_THRESHOLD = 0.50
VERDICTS = (
    (0.75, "likely fine"),
    (0.50, "probably fine"),
    (0.25, "suspicious - worth a human look"),
    (0.00, "likely wrong"),
)


def verdict(score: float, threshold: float | None = None) -> str:
    if threshold is not None:
        return "flagged" if score < threshold else "ok"
    for cut, text in VERDICTS:
        if score >= cut:
            return text
    return VERDICTS[-1][1]


def parse_archs(value: str) -> tuple[str, ...]:
    if not value or value == "default":
        return DEFAULT_ARCHS
    archs = tuple(a.strip() for a in value.split(",") if a.strip())
    bad = [a for a in archs if a not in ARCHS]
    if bad:
        raise SystemExit(f"unknown architecture(s): {', '.join(bad)}\n"
                         f"choose from: {', '.join(ARCHS)}")
    return archs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Hand-drawing quality scorer (P(good) per hand crop).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = ap.add_argument_group("input (give at least one)")
    src.add_argument("--image", action="append", default=[], metavar="PATH",
                     help="a single hand crop; repeatable")
    src.add_argument("--dir", action="append", default=[], metavar="DIR",
                     help="a folder of hand crops; repeatable")
    src.add_argument("--full", action="append", default=[], metavar="PATH",
                     help="a whole illustration; hands are detected first "
                          "(requires --detector)")

    mdl = ap.add_argument_group("model")
    mdl.add_argument("--backend", default="torch", choices=("torch", "onnx"),
                     help="torch (default) or onnx (no torch needed)")
    mdl.add_argument("--model", default="default",
                     help=f"comma-separated architectures, default "
                          f"'{','.join(DEFAULT_ARCHS)}'; "
                          f"available: {', '.join(ARCHS)}")
    mdl.add_argument("--weights", default=None,
                     help="folder holding the .pt/.onnx files "
                          "(default: ./weights next to this script)")

    det = ap.add_argument_group("detection (only used by --full)")
    det.add_argument("--detector", default=None,
                     help="path to a YOLOv8 hand-detector .pt; not bundled")
    det.add_argument("--min-det", type=float, default=0.40,
                     help="drop boxes below this detector confidence "
                          "(default 0.40; lower = more recall, more junk)")

    out = ap.add_argument_group("output")
    out.add_argument("--threshold", type=float, default=None,
                     help="flag scores below this value instead of printing a "
                          "banded verdict; use the output of calibrate.py")
    out.add_argument("--json", dest="json_path", default=None, metavar="PATH",
                     help="also write the full result list as JSON")
    args = ap.parse_args(argv)

    if not (args.image or args.dir or args.full):
        ap.error("give at least one of --image / --dir / --full")
    if args.full and not args.detector:
        ap.error("--full needs --detector (the hand detector is not bundled); "
                 "crop the hands yourself and use --image instead")

    # Resolve the default against this file, not the caller's cwd, so that
    # `python /path/to/predict.py` works from anywhere.
    weights_dir = Path(args.weights) if args.weights else Path(__file__).resolve().parent / "weights"
    backend = make_backend(args.backend, parse_archs(args.model), weights_dir)
    print(f"[model] {backend.kind} backend, ensemble of "
          f"{'+'.join(backend.archs)} ({len(backend.archs)} nets)", flush=True)

    records: list[dict] = []
    scores: list[float] = []

    # ---- 1. crops given directly ----
    crop_paths: list[Path] = [Path(p) for p in args.image]
    for d in args.dir:
        folder = Path(d)
        if not folder.is_dir():
            raise SystemExit(f"not a folder: {folder}")
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
            crop_paths += sorted(folder.glob(ext))

    crops, crop_sources = [], []
    for path in crop_paths:
        img = imread(path)
        if img is None:
            print(f"[skip] cannot read {path}")
            continue
        crops.append(img)
        crop_sources.append(str(path))

    if crops:
        for source, score in zip(crop_sources, backend.score(crops)):
            scores.append(score)
            records.append({
                "source": source,
                "hand_index": None,
                "bbox": None,
                "det_conf": None,
                "score": round(score, 4),
                "verdict": verdict(score, args.threshold),
            })

    # ---- 2. whole illustrations ----
    if args.full:
        from handqc.detector import HandDetector

        detector = HandDetector(args.detector)
        for path in args.full:
            img = imread(path)
            if img is None:
                print(f"[skip] cannot read {path}")
                continue
            hands = detector.detect(img, min_det=args.min_det)
            print(f"\n{Path(path).name}: {len(hands)} hand(s) above "
                  f"det>={args.min_det:.2f}")
            if not hands:
                print("  (no hands found — note the detector misses most "
                      "clenched-fist images, so this is not proof there are none)")
                continue
            for i, (hit, score) in enumerate(
                    zip(hands, backend.score([h["crop"] for h in hands])), 1):
                scores.append(score)
                flag = ""
                if hit["det_conf"] < 0.10 and score >= 0.5:
                    flag = "  <- conflict: low detection conf but high score"
                elif hit["det_conf"] < 0.50 and score >= 0.5:
                    flag = "  <- low detection conf (unusual pose, or junk box)"
                print(f"  #{i} bbox={hit['bbox']} det={hit['det_conf']:.2f} "
                      f"score={score:.4f}  {verdict(score, args.threshold)}{flag}")
                records.append({
                    "source": path,
                    "hand_index": i,
                    "bbox": list(hit["bbox"]),
                    "det_conf": round(hit["det_conf"], 3),
                    "score": round(score, 4),
                    "verdict": verdict(score, args.threshold),
                })

    if not records:
        print("\nNothing scored.")
        return 1

    # ---- 3. summary ----
    stats = summarize(scores)
    print(f"\n=== {stats['n']} hand(s) scored ===")
    print(f"  median {stats['median']:.3f}   mean {stats['mean']:.3f}   "
          f"p10 {stats['p10']:.3f}   p25 {stats['p25']:.3f}")
    if args.threshold is None:
        flagged = [r for r in records if r["score"] < 0.50]
        print(f"  below 0.50: {len(flagged)} "
              f"(the default cut is a starting point — run calibrate.py)")
    else:
        flagged = [r for r in records if r["score"] < args.threshold]
        print(f"  flagged at <{args.threshold:.3f}: {len(flagged)}")

    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(
            json.dumps({"summary": stats, "threshold": args.threshold,
                        "model": list(backend.archs),
                        "backend": backend.kind, "results": records},
                       ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"[out] {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
