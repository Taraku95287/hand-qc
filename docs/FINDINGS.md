# Five findings worth more than the AUC

The classifier is the least interesting thing this project produced. These are
the results that changed what I thought the task was, each with the measurement
behind it. Numbers are reproduced from `docs/RESULTS.md` /
`scripts/evaluate.py`.

---

## 1. The labels barely survived being re-derived

The whole project rests on 862 hand-labelled crops. So I tested whether the
`bad` labels were reproducible: the same hands, fresh pass, no access to the
original verdicts, three levels (`fine` / `questionable` / `wrong`) and — for
`wrong` — a named criterion out of five.

| outcome for the 186 `bad` labels | share |
|---|---:|
| reproducible as a concrete structural error | **2%** |
| judged `fine` on the second pass | **81%** |
| agreement on the clean subset, Cohen's κ | **−0.003** |

κ ≈ 0 means the two passes agree no more than two random labellers would. The
model's 0.8664 AUC is a faithful reproduction of *one* pass's boundary. It is
not a measurement of whether hands are drawn correctly, and no amount of extra
training data would make it one.

**Consequence:** I stopped describing this as an error detector and started
describing it as a taste filter. The same re-check also gives the honest base
rate — objectively broken hands are about **2%** of real generated art, not the
"AI can't draw hands" number everyone assumes. Three rounds of trying to force
the error rate up (11 generation configs, 68 samples) produced **zero**
structurally broken hands on current anime checkpoints.

What it would take to fix the labelling: ≥3 independent annotators, a written
criterion per gesture, and reporting the agreement matrix instead of a single
accuracy. Not done here.

## 2. Choosing a backbone by CV score is a coin flip, and I proved it by accident

First experiment: `resnet34` was the best backbone (0.8057), `resnet18` second
(0.8156), so `resnet34` looked like the pick. Same code, same data, re-run after
seeding the augmentation RNG: 0.8107 vs 0.8330 — **the ranking inverted.**

The cause was an unseeded global `np.random` used for augmentation. Two runs of
identical code differed by **>0.01 AUC**, and one downstream metric (cross-model
false-positive rate) moved between **84% and 0%**.

With the seed fixed, four backbones land at 0.8107 / 0.8289 / 0.8330 / 0.8567 —
a 0.046 spread for architectures spanning 1.53 M to 21.30 M parameters.

**Consequence:** the +0.033 that took the shipped model from 0.8330 to 0.8664
came from *ensembling two families*, not from architecture selection. Any
single-run backbone comparison in this space should be assumed to be noise until
it is repeated across seeds. Every artifact in this repo carries `seed` and
`select` in its `.npz` for exactly this reason.

## 3. A fixed threshold measures calibration, not discrimination

For a while this repo claimed `resnet34` "memorised the Illustrious style"
because it flagged 84% of good hands on an unseen model family. It does not. On
a seeded re-run it flagged **0%**.

Both runs had the same discrimination. What differed was where the score
distribution landed: in run one, good hands from the unseen family had a median
score of 0.244; in run two, 1.000. A threshold fixed at 0.5 turns a *calibration*
difference into an apparent *capability* difference.

The fix: align every model to the same in-domain operating point (reject 15% of
in-domain good hands), then compare. After alignment, the worst model in the
table is 1.7× its in-domain false-positive rate and the shipped ensemble is
0.0×. Nothing memorises style.

**Consequence:** any cross-domain comparison with a fixed cut-off is measuring
luck. This is also why `predict.py` takes `--threshold` and `calibrate.py` exists
at all — and why the retraction is still visible in `HAND_QUALITY.md` §12.2b
rather than quietly deleted.

## 4. In-domain AUC is almost blind to the failure that matters

Adding 86 hand crops from real user artwork to the 776-crop training set:

| | before | after |
|---|---:|---:|
| in-domain OOF AUC | 0.8678 | 0.8664 (**−0.001**) |
| score shift on real good hands | −0.258 | +0.008 |
| false positives on real good hands | **29.8%** | **2.1%** |

The headline metric got *slightly worse* while the tool went from "flags three
of every ten good images" to usable. Cross-validated AUC on the training
distribution cannot see this, because the shift is in the input distribution,
not in the model.

**Consequence:** for a filter, the deployment-relevant number is the
false-positive rate on the target distribution, and it has to be measured on
that distribution. 10% more data from the right place did more than any
modelling change. It also means a single "AUC 0.87" on a resume or a model card
is nearly information-free without the shift measurement next to it.

## 5. Free signals beat the models trained to replace them

Roughly 40% of boxes the hand detector returns on third-party art are not hands.
I built two gating models before checking what was already available.

| hand vs not-hand, 480 labelled crops (168 non-hands) | AUC |
|---|---:|
| crop size | 0.5433 |
| CNN gate trained for the job (168 samples) | 0.7532 |
| multi-task hand head (larger set, 445 non-hands) | 0.6873 |
| **raw detector confidence** | **0.9117** |

The detector's own confidence dominated, for free: no second model, no extra
inference, no extra bytes. A hand detector's uncertainty *is* a hand/not-hand
classifier. At `--min-det 0.40` it blocks 47% of non-hands while losing 3.8% of
real hands.

A related failure from the same round: making the model 3-class
(`good / bad / not_hand`) collapsed the task it was supposed to improve, taking
good/bad AUC from 0.833 to **0.64–0.74**, because 65 not-hand samples carried a
class weight of 4.75 and stole representation from the main task. Adding a class
is not free, even when it looks more correct.

**Consequence:** before training a component, measure the cheapest available
signal for it. Twice here that step was skipped, and twice the trained model was
worse than something already on the table.

---

## What these add up to

The task as posed — *"detect wrong hands"* — is under-specified. The interesting
engineering is not the classifier, it is:

1. establishing that the target is measurable at all (finding 1),
2. separating model capability from calibration drift (findings 2 and 3),
3. measuring on the distribution that matters instead of the one you have
   (finding 4),
4. preferring free signals over trained ones until proven otherwise (finding 5).

A defensible tool came out anyway — 61 MB, 0.8664 OOF AUC, 2.1% false positives
on real artwork after recalibration — but it ships with `calibrate.py`, an honest
limitations section, and a stated scope of "triage filter", because of the above.
