# Model card — hand-qc

**English** | [简体中文](MODEL_CARD.zh-CN.md)

## Overview

| | |
|---|---|
| Task | binary classification of a hand crop: `good` (would not be flagged) vs `bad` (would be flagged) |
| Output | `P(good)` in [0, 1]; a **low** score is a flag |
| Architecture | probability average of `resnet18` (11.19 M) + `efficientnet_b0` (4.05 M) |
| Size | 61 MB, 15.24 M parameters total |
| Input | BGR image of one hand, letterboxed to 224 × 224, ImageNet normalization |
| Runtime | ~9 ms per hand on the test machine (batch of 32, GPU); CPU is slower but fine |
| Weights | `resnet18_handqc.{pt,onnx}`, `efficientnet_b0_handqc.{pt,onnx}`, CC BY 4.0 |
| Framework | PyTorch ≥ 2.0 (torchvision), or ONNX Runtime ≥ 1.16 |
| Trained | 5-fold `StratifiedKFold`, seed 42, folds keyed on `gesture × base model` |

## Intended use

- **Triage.** Rank or gate hand crops before a human looks at them, in a
  batch/QC pipeline for AI-generated anime illustration.
- **Consistency.** Apply one person's standard to a large volume of images
  without them looking at all of it.
- **Research baseline.** A cheap reference point for anyone building hand
  quality assessment; the OOF vectors are shipped so it can be compared against
  without reproducing the training.

## Out-of-scope use

- **Claiming a hand is objectively correct or incorrect.** The model reproduces
  one annotator's boundary, and that annotator's own test–retest agreement was
  κ ≈ 0. An "objective error detector" claim is not supported by anything here.
- **Overall image quality.** It scores hands only. Nothing about composition,
  faces, colour, or whether the image is worth publishing.
- **Real hands / photographs.** Trained exclusively on anime-style crops.
- **Medical, biomechanical, or animation-rigging judgement.** This is not an
  anatomy model.
- **Any safety-critical decision.** It is a heuristic filter with a 60%
  precision on its positive class.
- **Enforcing a style.** It learned what one person considers correct drawing in
  one style family; using it as an automated standard on other artists'
  output would encode that preference as a gate.

## Metrics

Cross-validated on 862 crops (676 good / 186 bad), no held-out test set.

| metric | value | reference point |
|---|---:|---|
| OOF AUC | 0.8664 | 0.5 chance |
| accuracy @ 0.50 | 83.2% | 78.4% labelling everything good |
| precision on `bad` @ 0.50 | 59.6% | — |
| recall on `bad` @ 0.50 | 68.3% | — |
| precision / recall on `bad` @ aligned 0.547 | 56.4% / 71.0% | — |
| false positives on real user artwork, recalibrated | 2.1% | 29.8% before recalibration |

Per-gesture AUC ranges 0.68 (clenched hand) to 0.92 (own hands together). The
`bad` class is not uniformly represented: one gesture has 4 positive samples.

## Training data

862 hand crops labelled by a single annotator.

- 518 from `rinAnimeBlendArblend_v20` (Illustrious-family), generated locally.
- 258 from `waiANIMA_v10Base10` (Anima).
- 86 cropped from real user-published artwork.
- 10 gesture tags plus `unknown` for the real-artwork crops.

Crops come from `hand_yolov8s` detections with the box expanded 1.2× and a 48 px
minimum side. **The crop images are not redistributed**; `training/manifest.csv`
and `training/oof_*.npz` are.

Augmentation: rotation ±12°, scale-crop, HSV, contrast, 15% grayscale.
**No horizontal flip** — mirroring a hand turns a left hand into a right one.

## Ethical considerations

- **Annotator bias is the model.** The single-annotator `bad` boundary, and the
  fact that it was not self-consistent, means the model inherits one person's
  taste. It was trained on anime illustration, which has its own strong
  stylistic conventions and its own history of marginalising non-normative
  bodies; a hand-quality gate trained on that convention will treat unusual but
  deliberate stylisation as error.
- **Do not gate human artists' work with it.** As a pre-publish filter for one's
  own generations it is useful. As an automated standard applied to other
  people's art it encodes a preference as a verdict.
- **Derived data.** The training crops are outputs of Stable Diffusion
  checkpoints; redistribution terms follow those checkpoints, which is part of
  why the crops are not shipped. See `docs/DATA_AND_WEIGHTS.md`.
- **No personal data.** No identifiable people, no scraped photographs. The 86
  real-artwork crops are from published illustrations, used for training only and
  not redistributed.

## Caveats

1. Single annotator, self-agreement κ ≈ 0 — the central caveat. See
   `docs/FINDINGS.md` #1.
2. `bad` class under-sampled (186 total, as low as 4 for one gesture).
3. Single seed; run-to-run variation in this recipe exceeds 0.01 AUC.
4. No untouched held-out set.
5. Threshold is distribution-specific — recalibrate after any change of model or
   style (`calibrate.py`).
6. The upstream hand detector misses 82% of images whose hands are all clenched
   fists, so absence of a report is not evidence of correctness.
