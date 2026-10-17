# Training

**English** | [简体中文](README.zh-CN.md)

`train.py` is the script that produced the shipped weights. Each run trains one
small CNN — ImageNet-pretrained backbone, single-logit head, `BCEWithLogitsLoss`
with `pos_weight`, AdamW, cosine LR, early stopping, 5-fold OOF AUC, optional ONNX
export. It is a **reproduction artifact**, not a framework: the algorithm is
unchanged from the experiment behind `docs/RESULTS.md`. That includes the
augmentation ranges and the deliberate absence of a horizontal flip — mirroring a
hand changes which hand it is, so a flipped sample no longer matches its label.

The shipped scorer is two architectures, `resnet18` + `efficientnet_b0`, averaged
in probability space (see `handqc/backends.py`).

## The manifest

`manifest.csv` is the whole label set: **862 crops, 676 good / 186 bad**.

| column | meaning |
|---|---|
| `label` | `good` or `bad` — the only column trained on |
| `hand_id` | crop identifier; the image file is `<hand_id>.png` |
| `hand_tag` | gesture, 11 distinct values (`open hand, spread fingers`, `peace sign`, `unknown`, …) |
| `model` | source family: `IL` 518, `Anima` 258, `user` 86 (real ComfyUI output) |
| `num`, `det_conf` | row id, hand-detector confidence at crop time (unused by training) |

There is no `path` column, because **the crop images themselves are not
redistributed with this repository**. Supply them yourself:

```bash
--image-root /path/to/crops     # must contain every <hand_id>.png from the manifest
```

A manifest that *does* carry a `path` column is accepted too; that column wins and
`--image-root` becomes optional.

## Reproducing the shipped model

The defaults are the settings that were used (`--seed 42 --size 224 --epochs 30
--bs 32 --lr 3e-4 --folds 5 --select best`):

```bash
python training/train.py --image-root /path/to/crops --arch resnet18
python training/train.py --image-root /path/to/crops --arch efficientnet_b0
cp runs/resnet18_handqc.pt runs/efficientnet_b0_handqc.pt weights/   # names predict.py expects
```

(`Copy-Item` on PowerShell.) `--out` defaults to `runs/` and receives
`<tag>_handqc.pt`, `<tag>_handqc.onnx`, `oof_<tag>.npz`, plus
`cross_pred_<tag>.json` when `--cross` is given. `<tag>` is `<arch>`, suffixed with
`_s<size>` when `--size` is not 224 and with `_last` under `--select last`, so
experiments cannot overwrite each other.

Recorded OOF AUC at these settings: efficientnet_b0 0.8567, resnet18 0.8330,
mobilenet_v3_small 0.8289, resnet34 0.8107, ensemble 0.8664. Those gaps are about
the size of seed noise — two runs on the same data differ by ~0.01 OOF AUC — so the
ordering is not a result and was not used to pick the shipped pair.

## Runtime

Roughly 10 minutes per architecture (5 folds × 30 epochs at 224 px, early stopping
usually ending a fold before the budget) on the 8 GB GPU this was developed on; the
whole four-architecture table took ~40 minutes. `--size 320` roughly doubles that.
A CPU fallback exists but was never benchmarked. Never run two GPU jobs on the same
card — this project hit silent driver resets (TDR) that way.

## The caveat that matters most

The labels come from a single annotator. A blind re-annotation of the same crops by
an independent evaluator reproduced almost none of the good/bad calls: agreement was
100% on the crops marked good but 8% on those marked bad, and on the clean subset
Cohen's kappa was **−0.003** — zero agreement. So an AUC from this script is not
"how well the model detects anatomical errors"; it is how closely it reproduces one
person's boundary for what counts as a wrong hand, and that boundary is itself not
reproducible. Anything trained on these labels inherits that limit. The full record
is in `docs/RESEARCH_LOG.md` §13 and §16.
