#!/usr/bin/env python
"""Download the released weights into ./weights/.

The .pt/.onnx files are published as GitHub Release assets rather than
committed to git (61 MB of binaries do not belong in a repository history).

    python scripts/fetch_weights.py                    # everything
    python scripts/fetch_weights.py --onnx-only        # skip the .pt files
    python scripts/fetch_weights.py --arch resnet18    # one member model
    python scripts/fetch_weights.py --url-base https://.../v0.1.0

Set HANDQC_RELEASE_URL to override the default download location.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"

DEFAULT_URL_BASE = os.environ.get(
    "HANDQC_RELEASE_URL",
    "https://github.com/Taraku95287/hand-qc/releases/download/v0.1.0",
)

ARCHS = {
    "resnet18": "resnet18_handqc",
    "efficientnet_b0": "efficientnet_b0_handqc",
    "resnet34": "resnet34_handqc",
    "mobilenet_v3_small": "mobilenet_v3_small_handqc",
}
# Shipped artifacts per architecture. The first two are the ensemble members.
DISTRIBUTED = ("resnet18", "efficientnet_b0")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  GET {url}")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                pct = 100 * done / total
                print(f"\r    {done / 1e6:6.1f} / {total / 1e6:.1f} MB "
                      f"({pct:5.1f}%)", end="", flush=True)
    print()
    tmp.replace(dest)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fetch released model weights.")
    ap.add_argument("--url-base", default=DEFAULT_URL_BASE,
                    help="release asset base URL")
    ap.add_argument("--arch", action="append", default=None,
                    help=f"only this architecture; one of {', '.join(ARCHS)}")
    ap.add_argument("--onnx-only", action="store_true",
                    help="skip the PyTorch .pt files")
    ap.add_argument("--pt-only", action="store_true",
                    help="skip the ONNX files")
    ap.add_argument("--force", action="store_true", help="re-download")
    args = ap.parse_args(argv)

    archs = tuple(args.arch) if args.arch else DISTRIBUTED
    unknown = [a for a in archs if a not in ARCHS]
    if unknown:
        raise SystemExit(f"unknown architecture(s): {', '.join(unknown)}")

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    exts = []
    if not args.onnx_only:
        exts.append(".pt")
    if not args.pt_only:
        exts.append(".onnx")

    print(f"[out] {WEIGHTS}")
    failures = 0
    for arch in archs:
        for ext in exts:
            name = ARCHS[arch] + ext
            dest = WEIGHTS / name
            if dest.exists() and not args.force:
                print(f"  skip {name} ({dest.stat().st_size / 1e6:.1f} MB, "
                      f"sha256 {sha256_of(dest)[:12]}…)")
                continue
            try:
                download(f"{args.url_base.rstrip('/')}/{name}", dest)
                print(f"  ok   {name}  sha256 {sha256_of(dest)[:12]}…")
            except Exception as exc:
                failures += 1
                print(f"  FAIL {name}: {exc}")

    if failures:
        print(f"\n{failures} file(s) failed. If the release URL is not set yet, "
              f"pass --url-base or place the files in ./weights manually.")
        return 1
    print("\nDone. Try:  python predict.py --image examples/good_01.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
