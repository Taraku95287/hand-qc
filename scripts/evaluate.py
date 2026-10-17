#!/usr/bin/env python
"""Regenerate every number in docs/RESULTS.md from the shipped out-of-fold vectors.

    python scripts/evaluate.py
    python scripts/evaluate.py --json out/eval.json

The images are not redistributed, but the cross-validated predictions are:
`training/oof_<arch>.npz` holds the out-of-fold P(good) for all 862 crops plus
the fold assignment and the seed. Everything below is a pure function of those
vectors and `training/manifest.csv`, so the tables are checkable without the
dataset.

What is *not* reproducible here: anything that needed the cross-model test set
or the real-user crops. Those rows in RESULTS.md are marked accordingly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
TRAINING = ROOT / "training"
ARCHS = ("resnet18", "efficientnet_b0", "resnet34", "mobilenet_v3_small")


# ---------------------------------------------------------------- metrics
def auc(scores, labels) -> float:
    """Rank-based AUC (Mann-Whitney U), tie-aware."""
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels, np.int64)
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # average ranks over ties
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def confusion(scores, labels, threshold: float) -> dict:
    s = np.asarray(scores, np.float64)
    y = np.asarray(labels, np.int64)
    pred_bad = s < threshold              # low score -> flagged as bad
    tp = int(((pred_bad == 1) & (y == 0)).sum())
    fn = int(((pred_bad == 0) & (y == 0)).sum())
    fp = int(((pred_bad == 1) & (y == 1)).sum())
    tn = int(((pred_bad == 0) & (y == 1)).sum())

    def _div(a, b):
        return a / b if b else float("nan")

    p_bad, r_bad = _div(tp, tp + fp), _div(tp, tp + fn)
    f1_bad = _div(2 * p_bad * r_bad, p_bad + r_bad) if p_bad == p_bad else float("nan")
    return {
        "threshold": round(threshold, 4),
        "accuracy": round(_div(tp + tn, tp + tn + fp + fn), 4),
        "precision_good": round(_div(tn, tn + fn), 4),
        "recall_good": round(_div(tn, tn + fp), 4),
        "precision_bad": round(p_bad, 4),
        "recall_bad": round(r_bad, 4),
        "f1_bad": round(f1_bad, 4),
        "confusion": {"tp_bad": tp, "fn_bad": fn, "fp_bad": fp, "tn_bad": tn},
    }


def aligned_threshold(scores, labels, target_fpr: float = 0.15) -> float:
    """Working point that rejects `target_fpr` of the in-domain *good* hands.

    Fixed 0.5 confounds discrimination with calibration: an architecture whose
    scores happen to sit lower looks worse than it is. Aligning on a quantile of
    the good-hand distribution removes that.
    """
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels, np.int64)
    return float(np.quantile(scores[labels == 1], target_fpr))


# ---------------------------------------------------------------- loading
def load_manifest() -> list[dict]:
    path = TRAINING / "manifest.csv"
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_oof(arch: str) -> dict:
    path = TRAINING / f"oof_{arch}.npz"
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=True)
    return {
        "oof": z["oof"],
        "y": z["y"],
        "fold": z["fold"],
        "seed": int(z["seed"]),
        "select": str(z["select"]),
        "n": int(z["oof"].shape[0]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", dest="json_path", default=None,
                    help="also write every table as JSON")
    ap.add_argument("--fpr", type=float, default=0.15,
                    help="target false-positive rate used for threshold "
                         "alignment (default 0.15)")
    args = ap.parse_args(argv)

    manifest = load_manifest()
    labels = np.array([1 if r["label"] == "good" else 0 for r in manifest], np.int64)
    gestures = [r["hand_tag"] for r in manifest]
    models = [r["model"] for r in manifest]

    print(f"manifest: {len(manifest)} crops  "
          f"({int((labels == 1).sum())} good / {int((labels == 0).sum())} bad)")

    single = {}
    for arch in ARCHS:
        data = load_oof(arch)
        if not data:
            print(f"  [skip] no oof vector for {arch}")
            continue
        if data["n"] != len(manifest):
            raise SystemExit(f"{arch}: {data['n']} oof rows vs "
                             f"{len(manifest)} manifest rows — refusing to guess "
                             f"the row order")
        if not np.array_equal(data["y"], labels):
            raise SystemExit(
                f"{arch}: the labels in oof_{arch}.npz disagree with "
                f"manifest.csv. The row order must match; do not reorder the "
                f"manifest.")
        single[arch] = data

    if not single:
        raise SystemExit("no usable oof vectors found in training/")

    # ---- single models ----
    print("\n## Single models (5-fold OOF, seed 42)\n")
    print("| arch | seed | select | OOF AUC | fold std |")
    print("|---|---:|---|---:|---:|")
    table_single = []
    for arch, data in single.items():
        fold_aucs = [auc(data["oof"][data["fold"] == f], labels[data["fold"] == f])
                     for f in np.unique(data["fold"])]
        row = {
            "arch": arch,
            "seed": data["seed"],
            "select": data["select"],
            "auc": round(auc(data["oof"], labels), 4),
            "fold_std": round(float(np.std([a for a in fold_aucs if a == a])), 4),
        }
        table_single.append(row)
        print(f"| `{arch}` | {data['seed']} | {data['select']} | "
              f"**{row['auc']:.4f}** | {row['fold_std']:.4f} |")

    # ---- ensemble ----
    members = [a for a in ("resnet18", "efficientnet_b0") if a in single]
    if len(members) < 2:
        print("\n(ensemble needs resnet18 + efficientnet_b0)")
        return 1
    ens = np.mean([single[a]["oof"] for a in members], axis=0)
    base_auc = auc(ens, labels)
    print(f"\n## Ensemble {'+'.join(members)} (probability average)\n")
    print(f"OOF AUC **{base_auc:.4f}**\n")

    thr_aligned = aligned_threshold(ens, labels, args.fpr)
    print(f"## Confusion at several operating points\n")
    print(f"`bad` is the positive class throughout: a low score is a flag.\n")
    print("| threshold | accuracy | P(good) | R(good) | P(bad) | R(bad) | F1(bad) |")
    print("|---|---|---|---|---|---|---|")
    rows_conf = []
    for name, thr in (("0.40", 0.40),
                      ("0.50", 0.50),
                      (f"{thr_aligned:.3f} (aligned, {args.fpr:.0%} FPR)", thr_aligned)):
        c = confusion(ens, labels, thr)
        c["label"] = name
        rows_conf.append(c)
        print(f"| {name} | {c['accuracy']:.3f} | {c['precision_good']:.3f} | "
              f"{c['recall_good']:.3f} | {c['precision_bad']:.3f} | "
              f"{c['recall_bad']:.3f} | {c['f1_bad']:.3f} |")

    majority = float((labels == 1).mean())
    print(f"\nDo-nothing baseline (label everything good): accuracy {majority:.3f}, "
          f"and it finds none of the {int((labels == 0).sum())} bad hands.")

    # ---- per gesture ----
    print("\n## Per-gesture AUC\n")
    print("| gesture | n | good | bad | AUC |")
    print("|---|---:|---:|---:|---:|")
    rows_gesture = []
    for g in sorted(set(gestures)):
        idx = np.array([x == g for x in gestures])
        a = auc(ens[idx], labels[idx])
        row = {"gesture": g, "n": int(idx.sum()),
               "good": int((labels[idx] == 1).sum()),
               "bad": int((labels[idx] == 0).sum()),
               "auc": None if a != a else round(a, 4)}
        rows_gesture.append(row)
        auc_text = "n/a" if row["auc"] is None else f"{row['auc']:.3f}"
        print(f"| {g} | {row['n']} | {row['good']} | {row['bad']} | {auc_text} |")
    rows_gesture.sort(key=lambda r: (r["auc"] is None, r["auc"] or 0))

    # ---- per base model ----
    print("\n## Per base model\n")
    print("| base model | n | good | bad | AUC |")
    print("|---|---:|---:|---:|---:|")
    rows_model = []
    for m in sorted(set(models)):
        idx = np.array([x == m for x in models])
        a = auc(ens[idx], labels[idx])
        row = {"model": m, "n": int(idx.sum()),
               "good": int((labels[idx] == 1).sum()),
               "bad": int((labels[idx] == 0).sum()),
               "auc": None if a != a else round(a, 4)}
        rows_model.append(row)
        auc_text = "n/a" if row["auc"] is None else f"{row['auc']:.3f}"
        print(f"| {m} | {row['n']} | {row['good']} | {row['bad']} | {auc_text} |")

    if args.json_path:
        payload = {
            "n": len(manifest),
            "good": int((labels == 1).sum()),
            "bad": int((labels == 0).sum()),
            "n_folds": int(len(np.unique(single[members[0]]["fold"]))),
            "single_models": table_single,
            "ensemble": {"members": members, "auc": round(base_auc, 4)},
            "confusion": rows_conf,
            "majority_baseline_accuracy": round(majority, 4),
            "per_gesture": rows_gesture,
            "per_model": rows_model,
            "fpr_target": args.fpr,
            "aligned_threshold": round(thr_aligned, 4),
        }
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"\n[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
