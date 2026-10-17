# hand-qc — flag badly drawn hands in anime illustrations

A small classifier that scores a **hand crop** and answers one question: *does
this hand look drawn correctly?* Built as a pre-publish filter for AI-generated
illustrations, where broken hands are the single most common defect.

- **61 MB**, 15.2 M parameters, no CLIP, no foundation model, runs on CPU.
- **OOF AUC 0.8664** on 862 human-labelled hand crops, cross-validated with the
  folds stratified by *gesture × base model* so neither can be memorised.
- Ships **PyTorch and ONNX** weights plus the **out-of-fold predictions**, so
  every number in `docs/RESULTS.md` is reproducible without the image data.

```bash
python scripts/fetch_weights.py
python predict.py --image examples/good_02.png      # -> score 0.9982
python predict.py --image examples/bad_03.png       # -> score 0.1371
```

---

## What this thing actually does

Given a hand crop it returns `P(good)`, where "good" means *a human would not
flag this hand*. It was trained on 862 crops the author labelled by hand and it
reproduces that judgement — nothing more.

**It is not an "anatomically correct hand" detector.** Two measurements forced
that sentence into the README:

1. A blind three-level re-check of the author's own `bad` labels found only
   **2%** were reproducible as an objective structural error; **81%** were
   judged fine by a second pass. Agreement on the clean subset was
   **Cohen's κ ≈ 0 (−0.003)**.
2. Re-labelling real user art, the objective error rate came out at **~2%** —
   far below what "AI hands are always broken" would predict.

So the honest framing is: this is a **cheap, consistent re-implementation of one
person's taste**, useful as a triage filter, and the interesting part of the
project turned out to be *measuring how ill-defined the task is* rather than
chasing AUC. `docs/FINDINGS.md` has the five results worth reading.

## Results

Ensemble = `resnet18 + efficientnet_b0`, probability average. `bad` is the
positive class: a **low** score is a flag.

| metric | value |
|---|---|
| out-of-fold AUC, 862 crops | **0.8664** |
| accuracy @ threshold 0.50 | 83.2% |
| precision / recall on `bad` @ 0.50 | 59.6% / 68.3% |
| precision / recall on `bad` @ 0.547 (aligned to 15% FPR) | 56.4% / 71.0% |
| do-nothing baseline (label everything good) | 78.4% accuracy, 0 bad found |
| false positives on **real user artwork**, after recalibration | **2.1%** |
| model size / params | 61 MB / 15.24 M |
| scoring throughput | ~9 ms per hand |

Per-gesture AUC spans **0.68** (clenched hand) to **0.92** (own hands together)
— a 0.24 spread that a single headline number hides. Full tables, the
cross-model evaluation and the ablations are in **`docs/RESULTS.md`**;
`python scripts/evaluate.py` regenerates the core tables from the shipped OOF
vectors.

### Baselines it beats and loses to

| approach | size | OOF AUC |
|---|---|---|
| CLIP ViT-H/14 linear probe ¹ | 2411 MB | 0.7709 |
| CLIP ViT-H/14 fine-tuned ¹ | 2411 MB | 0.7957 |
| resnet18 alone | 45 MB | 0.8330 |
| efficientnet_b0 alone | 16 MB | 0.8567 |
| **shipped ensemble** | **61 MB** | **0.8664** |
| three-model ensemble | 67 MB | 0.8706 |

¹ Measured on the earlier 776-crop set; the CLIP runs were not repeated on the
final 862. The small-model rows are all 862.

The gap from 0.8330 to 0.8664 came from ensembling, not from architecture
shopping: the four backbones' solo scores (0.811–0.857) differ by roughly the
seed-to-seed noise, so picking one by a single CV number is a coin flip.

## Install

```bash
pip install -r requirements.txt          # numpy, opencv, torch, torchvision
python scripts/fetch_weights.py          # 61 MB into ./weights/
```

CPU-only, and no CUDA needed:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

No torch at all? Use the ONNX path:

```bash
pip install numpy opencv-python onnxruntime
python scripts/fetch_weights.py --onnx-only
python predict.py --image examples/good_02.png --backend onnx
```

ONNX and PyTorch agree to **8.9e-07** max absolute difference over 862 crops.

## Usage

```bash
# one crop
python predict.py --image examples/good_02.png

# a folder
python predict.py --dir examples/ --json out/scores.json

# a whole illustration: detect hands first, then score each
python predict.py --full art.png --detector weights/hand_yolov8s.pt
```

`--full` needs a YOLOv8 hand detector, which is **third-party and not bundled**
(`ultralytics` is AGPL-3.0 — see `LICENSE-MODEL.md`). Without one, crop the
hands yourself and use `--image` / `--dir`.

### You must calibrate the threshold before trusting it

A threshold fitted in the training domain raised the false-positive rate on
genuinely good real-artwork hands to **29.8%** — three good images in ten
flagged. Adding 86 real-distribution crops to training brought that to **2.1%**,
but any change of model, style or rendering pipeline can shift the score
distribution again.

`calibrate.py` re-fits the threshold **on your own images, with no labels**, by
assuming most of your hands are fine (objective error rate is ~2%) and taking
the quantile that rejects `--target-fpr` of them:

```bash
python calibrate.py --dir my_renders/ --detector weights/hand_yolov8s.pt --target-fpr 0.10
python predict.py --full new_art.png --detector weights/hand_yolov8s.pt --threshold 0.412
```

## How it works

```
illustration ──YOLOv8 hand detector──► crops (bbox ×1.2, min 48 px, det conf ≥ 0.40)
                                          │
                          resnet18 ───────┤
                       efficientnet_b0 ───┤──► probability average ──► P(good) ──► threshold
```

Two design choices that are easy to get wrong:

- **Preprocessing letterboxes, it does not squash.** Hands are not square, and
  stretching a crop changes the finger-length ratios the model is judging. The
  long side goes to 224 px and the short side is centre-padded.
- **No horizontal flip in augmentation.** Every other flip-free augmentation
  (rotation ±12°, scale, HSV, contrast, 15% grayscale) is used, but mirroring a
  hand converts a left hand into a right one and teaches the model that handedness
  is noise.

The detector's confidence is also used as a free out-of-distribution gate. On
480 labelled crops (168 confirmed non-hands) raw detection confidence separated
hand from not-hand at **AUC 0.9117**; a CNN gate trained on the same crops
reached 0.7532, and on a larger harder set it fell to 0.6873. On third-party
illustrations about 40% of detected boxes are not hands — lanterns, braids,
claws, flowers, trees — so that gate is not optional.

## Known limitations

- **It cannot see what the detector cannot find.** `hand_yolov8s` misses **82%**
  of images whose only hands are clenched fists. "0 hands found" is not evidence
  that the hands are fine.
- **Weak on exactly the cases people fight about.** Clenched hand (AUC 0.682)
  and holding-object (0.748) are the worst gestures, and those are the ones with
  ambiguous boundaries.
- **One annotator, no inter-annotator agreement.** The labels are the author's.
  See `docs/FINDINGS.md` #1 before quoting any accuracy number.
- **Small, narrow dataset.** 862 crops from two SD checkpoints plus 86 real-user
  crops. The `bad` class is under-sampled (186) and the `good` class is
  over-represented for gestures that are easy to draw.
- **Calibration is per-distribution.** Re-run `calibrate.py` when you change
  models or style.
- **Not a general-purpose quality metric.** It scores hands. It does not know
  whether the picture is any good.

## Repository layout

```
predict.py              CLI: score crops, or whole illustrations
calibrate.py            CLI: fit a threshold on your own unlabelled images
handqc/                 importable package
  preprocess.py         unicode-safe I/O, letterbox, normalization
  backends.py           PyTorch and ONNX scorers, single API
  detector.py           optional YOLOv8 hand detector + OOD gate
  calibration.py        quantile thresholding, no labels required
scripts/
  fetch_weights.py      pull the 61 MB of weights from the release
  evaluate.py           rebuild the RESULTS.md tables from shipped OOF vectors
training/
  train.py              reproduction of the training run
  manifest.csv          862 rows: label, gesture, base model, detector conf
  oof_*.npz             out-of-fold predictions + fold + seed
docs/
  RESULTS.md            every measured number, including the failures
  FINDINGS.md           the five results worth reading
  RESEARCH_LOG.md       the raw chronological log (1.6k lines, has retractions)
  DATA_AND_WEIGHTS.md   where the data came from and what is not redistributed
MODEL_CARD.md           intended use, out-of-scope use, ethical notes
```

## License

Code MIT (`LICENSE`). Weights CC BY 4.0 (`LICENSE-MODEL.md`).
`hand_yolov8s` is third-party and not redistributed. Training crops are not
redistributed either — see `docs/DATA_AND_WEIGHTS.md`.
