# Data and weights provenance

**English** | [简体中文](DATA_AND_WEIGHTS.zh-CN.md)

## What is in this repository

| artifact | shipped | notes |
|---|---|---|
| `training/manifest.csv` | ✅ 862 rows | `num, label, hand_id, hand_tag, model, det_conf` — no image paths, no pixels |
| `training/oof_<arch>.npz` | ✅ 4 files, 5 KB each | out-of-fold `P(good)`, label, fold, seed, selection mode |
| `training/train.py` | ✅ | the training run, path-externalised |
| `examples/*.png` | ✅ 8 crops | 4 good / 4 bad, for the quickstart |
| `weights/*.pt`, `weights/*.onnx` | GitHub Release | 61 MB for the shipped pair; fetched by `scripts/fetch_weights.py` |
| **hand crop images (862)** | ❌ | not redistributed |
| **source illustrations** | ❌ | not redistributed |
| **`hand_yolov8s.pt`** | ❌ | third-party model, not mine to redistribute |

The manifest + OOF vectors are shipped deliberately: they make every table in
`docs/RESULTS.md` reproducible (`python scripts/evaluate.py`) without handing out
a derived image set.

## How the training data was produced

1. **Generation.** Illustrations were generated locally with Stable Diffusion
   checkpoints, using a fixed prompt template per gesture so that gesture,
   composition and framing were held roughly constant across the dataset:
   - `rinAnimeBlendArblend_v20` — Illustrious-family, 518 crops
   - `waiANIMA_v10Base10` — Anima, 258 crops
   - 86 further crops were cut from real published artwork, to bring the real
     distribution into training (this is the fix described in
     `docs/RESULTS.md`, 29.8% → 2.1% false positives)
2. **Detection.** `hand_yolov8s` located hands; boxes were expanded 1.2×,
   minimum side 48 px, detection confidence ≥ 0.25 at harvest time.
3. **Labelling.** Each crop was labelled `good` or `bad` by one annotator, with
   the gesture tag and base-model tag recorded alongside. Gesture tags were
   verified to exist in the Danbooru tag vocabulary so that the prompt wording
   matched what the models were trained to respond to.
4. **Folds.** `StratifiedKFold(5)` over the string `model|hand_tag`, so that no
   fold contains a base model or a gesture that another fold has not seen. This
   blocks two shortcuts that would otherwise inflate every number: "Illustrious
   hands are bad" and "clenched hands are bad".

## Why the crop images are not redistributed

Two reasons, in order of importance:

1. **They are derived from third-party model output.** The crops are fragments of
   images produced by named Stable Diffusion checkpoints. Their terms are those
   checkpoints', and re-hosting a derived image set is not the author's call to
   make. The 86 crops from real published artwork are more clearly not
   redistributable.
2. **They are not needed.** Everything quantitative is a function of the manifest
   plus the OOF vectors, both of which are shipped.

`examples/` contains 8 crops purely so that a reader can verify a fresh install
works. They are labelled in `examples/labels.csv`.

## The hand detector

`--full` mode needs a YOLOv8 hand detector. It is **not** included:

- It is a third-party model.
- `ultralytics`, its runtime, is **AGPL-3.0**. Importing it in this repo is fine;
  redistributing an application that links against it is a licensing decision you
  should make deliberately, not inherit by accident.

Point `--detector` at your own copy. Everything except `--full` works without it:
`--image` and `--dir` take crops directly.

**Known blind spot, measured:** `hand_yolov8s` misses **82%** of images whose only
hands are clenched fists. Its training data apparently contained very few. If a
report says "0 hands found", that is not evidence the hands are fine.

## Weights

| architecture | params | `.pt` | `.onnx` |
|---|---:|---:|---:|
| `resnet18` | 11.19 M | 44.8 MB | 44.7 MB |
| `efficientnet_b0` | 4.05 M | 16.3 MB | 16.0 MB |
| `mobilenet_v3_small` | 1.53 M | 6.2 MB | 6.1 MB |
| `resnet34` | 21.30 M | 85.3 MB | 85.2 MB |

The first two are the shipped ensemble. The other two are optional and are not
used by default: 0.8289 and 0.8107 OOF AUC respectively, both below either
shipped member.

ONNX export used `torch.onnx.export(..., dynamo=False, opset_version=14)`.
Verified against the PyTorch path on the 8 example crops and on the full set:
max absolute difference **8.9e-07** (CPU). On CUDA the PyTorch path uses TF32
convolutions, which widens that to ~8.5e-04 — that is the hardware's default
precision, not an export error.

## Reproducing the training run

```bash
cd training
python train.py --manifest manifest.csv --image-root /path/to/your/crops \
    --arch resnet18 --seed 42
```

You need the crop images, which are not shipped. See `training/README.md` for the
expected layout and for the honest caveat about the labels.
