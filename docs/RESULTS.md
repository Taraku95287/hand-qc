# Results

Every number here was measured on this machine; nothing is estimated. Where a
result was later retracted or contradicted, the retraction is in the same
section rather than deleted — `docs/RESEARCH_LOG.md` has the full chronology.

## How to reproduce

The core tables are a pure function of the shipped out-of-fold vectors, so they
can be rebuilt **without the image data**:

```bash
python scripts/evaluate.py
```

That reads `training/manifest.csv` (862 rows) and `training/oof_*.npz`
(cross-validated `P(good)`, fold index, seed). It refuses to run if the row
order of the manifest and the OOF vectors disagree, because a silent mismatch
would produce plausible-looking nonsense.

Sections marked **[not reproducible here]** needed the cross-model test set or
the real-user crops, which are not redistributed.

## Dataset

862 hand crops, each one human-labelled `good` / `bad`:

| | count |
|---|---:|
| total | 862 |
| `good` (would not flag) | 676 |
| `bad` (would flag) | 186 |
| base model: Illustrious (`rinAnimeBlendArblend_v20`) | 518 |
| base model: Anima (`waiANIMA_v10Base10`) | 258 |
| real user artwork (86 images) | 86 |
| distinct gesture tags | 10 + `unknown` |

The 10 gestures: `clenched hand`, `v sign`, `open hand, spread fingers`,
`holding object`, `own hands together`, `pointing`, `thumbs up`,
`hand on own chest`, `outstretched hand`, `peace sign`.

The `bad` class is the scar of this project: it never got large enough. Three
rounds of targeted generation (11 configs, 68 samples) produced **zero**
structurally broken hands, because current anime checkpoints simply do not fail
the way SD 1.5 did. See `docs/FINDINGS.md` #1.

## Headline

Ensemble `resnet18 + efficientnet_b0`, probability average, seed 42,
`StratifiedKFold(5)` with folds keyed on `model|hand_tag`.

**Out-of-fold AUC 0.8664.**

### Confusion at several operating points

`bad` is the positive class: a low score is a flag.

| threshold | accuracy | P(good) | R(good) | P(bad) | R(bad) | F1(bad) |
|---|---:|---:|---:|---:|---:|---:|
| 0.40 | 0.838 | 0.889 | 0.905 | 0.632 | 0.591 | 0.611 |
| **0.50** | **0.832** | **0.909** | **0.873** | **0.596** | **0.683** | **0.637** |
| 0.547 (aligned to 15% FPR) | 0.819 | 0.914 | 0.849 | 0.564 | 0.710 | 0.629 |

Counts at 0.50: 127 of 186 `bad` caught, 59 missed, 86 of 676 `good` wrongly
flagged.

**Read the baseline before the accuracy.** Labelling everything `good` scores
78.4% — so 83.2% is a 4.8-point gain, not a 83% solution. The metric that
matters for a filter is `P(bad)` 0.596: when it flags a hand, it is right about
three times in five.

## Per-gesture AUC

| gesture | n | good | bad | AUC |
|---|---:|---:|---:|---:|
| own hands together | 81 | 55 | 26 | 0.923 |
| open hand, spread fingers | 130 | 61 | 69 | 0.849 |
| hand on own chest | 72 | 58 | 14 | 0.836 |
| outstretched hand | 82 | 59 | 23 | 0.797 |
| v sign | 56 | 51 | 5 | 0.796 |
| peace sign | 92 | 80 | 12 | 0.785 |
| pointing | 66 | 52 | 14 | 0.780 |
| holding object | 67 | 60 | 7 | 0.748 |
| thumbs up | 71 | 59 | 12 | 0.727 |
| clenched hand | 59 | 55 | 4 | **0.682** |
| `unknown` (real user crops) | 86 | 86 | 0 | n/a |

A 0.24 AUC spread. The two worst — clenched hand and holding-object — are
exactly the gestures where "what counts as wrong" is most arguable, which is
consistent with the labelling finding rather than a coincidence.

## Per base model

| base model | n | good | bad | AUC |
|---|---:|---:|---:|---:|
| Anima | 258 | 216 | 42 | 0.939 |
| Illustrious | 518 | 374 | 144 | 0.807 |

The Anima subset is easier *and* better labelled: it has fewer `bad` samples per
good sample, and its bad ones came from a narrower failure mode. Do not read
this as "the model understands Anima better."

## Single models and ensembles

All rows below are 862 crops, seed 42, same recipe.

| model | params | size | OOF AUC |
|---|---:|---:|---:|
| `resnet34` | 21.30 M | 85 MB | 0.8107 |
| `mobilenet_v3_small` | 1.53 M | 6 MB | 0.8289 |
| `resnet18` | 11.19 M | 45 MB | 0.8330 |
| `efficientnet_b0` | 4.05 M | 16 MB | 0.8567 |
| `resnet18 + mobilenet_v3_small` | 12.72 M | 51 MB | 0.8526 |
| **`resnet18 + efficientnet_b0`** | **15.24 M** | **61 MB** | **0.8664** |
| `resnet18 + efficientnet_b0 + mobilenet_v3_small` | 16.77 M | 67 MB | 0.8706 |

Ensembling buys more than architecture choice: the four backbones span
0.811–0.857 for the same recipe, and that spread is roughly seed noise (see
below), while adding a second family is worth +0.03.

The three-model ensemble is 0.004 AUC better for 6 MB more. The shipped default
stays at two models; it is one constant (`DEFAULT_ARCHS` in
`handqc/backends.py`).

### Rank averaging vs probability averaging

Rank averaging scored 0.8705 in-domain against 0.8664 for probability
averaging — but rank averaging needs a batch to rank within, and a deployed
scorer sees one crop. **Deployed numbers are probability averages.**

### Why not pick a backbone by CV score

The first run of this experiment ranked the backbones `resnet34` > `resnet18`.
After fixing the augmentation RNG seed, the same code and data produced
`resnet34` 0.8261 / `resnet18` 0.8387 — the ranking flipped. The cause was a
global, unseeded `np.random` in augmentation; individual runs differed by more
than 0.01 AUC and cross-model false-positive rates moved between 84% and 0%.
Every number on this page comes from a `--seed 42` run.

## Threshold alignment — the mistake that looked like a discovery

Measured with a **fixed 0.5** threshold, `resnet34` flagged 84% of good hands on
an unseen model family, which read as "it memorised the Illustrious style". With
a fixed seed it became **0%**.

Nothing about the architecture had changed. Its score distribution on the
unseen family had collapsed below 0.5 in one run and not in the other: that is a
calibration failure, not a discrimination failure, and a fixed threshold
measures both at once.

The fix is to align each model to the same in-domain working point — reject 15%
of in-domain good hands — and then ask how many good hands it rejects
out-of-domain:

| model | aligned threshold | in-domain `bad` caught | cross-model FPR@0.5 | cross-model FPR (aligned) |
|---|---:|---:|---:|---:|
| `resnet18` | 0.424 | 58.6% | 1.3% | 0.0% |
| `efficientnet_b0` | 0.497 | 65.6% | 2.7% | 2.7% |
| `mobilenet_v3_small` | 0.590 | 60.8% | 1.3% | 1.3% |
| `resnet34` | 0.316 | 60.8% | 25.3% | 25.3% |
| **`resnet18 + efficientnet_b0`** | **0.547** | **71.0%** | 0.0% | **0.0%** |

## Cross-model transfer **[not reproducible here]**

75 hands cropped from FLUX.2 illustrations; the model family never appeared in
training. All 75 were `good` — a cross-model *AUC* was impossible, so the metric
is false-positive rate instead.

| quantity | value |
|---|---|
| score shift, ensemble (median good − median good in-domain) | **+0.026** |
| false positives at the aligned threshold | **0.0%** |

**The scope of this result is narrow and it was overstated once already.** The
test kept the author's prompt template, composition and framing; only the
checkpoint changed. The next section is what happens when everything changes.

## Real-world distribution **[not reproducible here]** — the 29.8% → 2.1% fix

Applied to real user artwork with the training-domain threshold (0.509):

| quantity | before | after |
|---|---:|---:|
| score shift on real good hands | **−0.258** | **+0.008** |
| false positives on real good hands | **29.8%** (14/47) | **2.1%** |

The fix was not a new model. It was adding **86 hand crops from real user
images** to the 776-crop training set. In-domain AUC barely moved (0.8678 →
0.8664) while the deployment behaviour changed from unusable to usable.

Two lessons, both in `docs/FINDINGS.md`:
- In-domain AUC is nearly blind to distribution shift. The number that moved
  was the false-positive rate on real images, and nobody was measuring it until
  a real user's images were run through.
- 86 crops out of 862 — 10% — fixed it. The gap was not a modelling gap.

## Not-hand gating **[not reproducible here]**

On third-party illustrations roughly 40% of detected boxes are not hands
(lanterns, braids, claws, flowers, trees). Three candidate gates, measured on
480 hand/non-hand crops (168 confirmed non-hands):

| signal | AUC | cost |
|---|---:|---|
| crop size | 0.5433 | free |
| trained CNN gate, 65 non-hands | 0.8710 | +6 MB, overfits |
| trained CNN gate, 168 non-hands | 0.7532 | +6 MB |
| **raw detector confidence** | **0.9117** | **free** |

The detector's own confidence beat both purpose-trained gates. It is a hand
detector; when it is unsure, that usually means it is not looking at a hand.

Threshold sweep on the same 480 crops:

| `--min-det` | non-hands blocked | real hands lost | non-hands remaining |
|---|---:|---:|---:|
| 0.30 | 23.2% | 1.9% | 76.8% |
| 0.35 | 31.0% | 1.9% | 69.0% |
| **0.40 (default)** | **47.0%** | **3.8%** | **53.0%** |
| 0.45 | 53.0% | 6.4% | 47.0% |
| 0.50 | 63.7% | 9.3% | 36.3% |
| 0.60 | 79.8% | 12.5% | 20.2% |

On a larger, harder set (1124 crops, 445 non-hands) the trained gate fell to
**0.6873** and would have cost 10.9% of real hands at 0.5.

## Things that did not work

Recorded so the next person does not repeat them. Numbers from
`docs/RESEARCH_LOG.md` §19 unless noted.

| attempt | result |
|---|---|
| 3-class `good / bad / not_hand` jointly | main task collapsed from 0.833 to **0.64–0.74** AUC. 65 not-hands got a class weight of 4.75 and stole representation from the task that mattered. |
| two-head multi-task (shared trunk) | quality head 0.7437 AUC, hand head 0.6382 — worse than a standalone gate, and it cost 25% of real hands |
| hand skeleton (SDPose) on hand crops | **8% detection** (3 of 40). Whole-body model; cropping the body removes what it infers from |
| self-trained 21-keypoint skeleton model | requires 21 × 336 = 7056 pixel-accurate coordinates. Abandoned on cost |
| 5 inpainting-based degradation methods to synthesise `bad` hands | produced artefacts the model learned instead of anatomy |
| MediaPipe hands | unusable on anime line art |
| pairing a source image with its generated image | no signal; the pairing itself was the defect |
| CLIP ViT-H/14 probe / fine-tune | 0.7709 / 0.7957 at 2411 MB — see below |

### CLIP baseline

| approach | size | OOF AUC |
|---|---|---:|
| CLIP ViT-H/14 linear probe | 2411 MB + 1281 params | 0.7709 |
| CLIP ViT-H/14 fine-tuned | 2411 MB | 0.7957 |
| shipped small ensemble | 61 MB | 0.8664 |

Measured on the 776-crop training set (the CLIP runs were not repeated on the
final 862-crop set). The 61 MB model beats a 2.4 GB foundation model by ~0.07
AUC while being 40× smaller and needing no GPU. This is the reason the project
stopped trying to make CLIP work.

## Limitations of these numbers

1. **One annotator.** Cohen's κ against an independent blind re-check of the same
   `bad` labels was **−0.003**. The AUC measures agreement with one person's
   boundary, not anatomical correctness.
2. **`bad` is under-sampled.** 186 positives, unevenly spread: one gesture has 4.
   Per-gesture numbers with n<10 positives are noise.
3. **No held-out test set.** All numbers are cross-validated. There is no
   untouched partition that has never influenced a decision.
4. **Single seed.** The seed-sensitivity experiment above shows individual runs
   move by >0.01 AUC. Repeating the final recipe across ≥3 seeds and reporting
   mean ± std has not been done.
5. **Detection ceiling.** The detector misses 82% of images whose hands are all
   clenched fists, so recall on those images is zero for reasons unrelated to
   this model.
