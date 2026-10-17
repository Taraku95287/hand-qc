# -*- coding: utf-8 -*-
"""Train the small hand-quality classifier: the actual self-contained model.

Why this exists
---------------
The first "model" in this project was a CLIP ViT-H/14 (632M parameters,
2411 MB) with a logistic-regression head on top of it (1281 trainable
parameters). That is a linear probe, not an independent model: 0.0002% of the
pipeline was actually trained, and the 2.4 GB base checkpoint competed for VRAM
with the image generator on an 8 GB card.

This script trains a CNN that is a few tens of MB and runs on its own:

    ResNet-18          11.19M params   ~45 MB
    ResNet-34          21.30M params   ~85 MB
    EfficientNet-B0     4.05M params   ~16 MB
    MobileNetV3-Small   1.53M params    ~6 MB

(Counts are the measured ``sum(p.numel())`` of the shipped ``.pt`` files, which
is smaller than the published paper figures for the two EfficientNet-family
nets because both heads are replaced by a single logit.)

No CLIP checkpoint is loaded here and no external baseline is consulted: this
script trains small models only, initialised from ImageNet weights.

Training strategy (unchanged from the experiment this reproduces)
-----------------------------------------------------------------
* ImageNet-pretrained initialisation (weights are cached locally by torchvision;
  no network access is needed once they exist).
* Stratified K-fold, stratified on the string ``model|hand_tag``.
* Augmentation: +-12 deg rotation / 0.85-1.00 scale crop / HSV jitter /
  contrast / brightness / 15% grayscale.
  **No horizontal flip** -- mirroring a hand changes which hand it is, so a
  flip would corrupt the label semantics (see HandDS.__getitem__).
* Class weighting via ``pos_weight`` plus early stopping.
* ONNX export so the model can be deployed without PyTorch.

Reading the OOF AUC
-------------------
With a fixed seed on the 862-crop manifest this script records

    efficientnet_b0     0.8567
    resnet18            0.8330
    mobilenet_v3_small  0.8289
    resnet34            0.8107
    ensemble r18+b0     0.8664   <- what ships

Those four single-model numbers all sit within roughly one standard deviation
of seed noise (two runs of the *same* data differ by ~0.01 OOF AUC), so a single
OOF number is not a basis for choosing an architecture, and the ordering above
has already flipped once between runs. What actually moved the number was more
data and ensembling, not architecture selection.

The labels come from one annotator. A blind re-annotation of the same crops
reproduced almost none of the good/bad calls (Cohen's kappa ~= 0 on the clean
subset), so every AUC printed below measures agreement with that annotator's
boundary, not objective anatomical correctness. See docs/RESEARCH_LOG.md.

Cross-model caveat (a retracted conclusion, kept as a warning)
--------------------------------------------------------------
An early version of this project claimed ResNet-34 had "memorised the IL/Anima
style" because it flagged 84% of good hands on an unseen model family. Rerun
with a fixed seed, the same architecture measured 0.0%. The 84% was calibration
collapse -- the whole score distribution shifts out of domain -- not a
discrimination failure, and the number swings wildly with the seed. A
false-positive rate at a *fixed* threshold therefore cannot be compared across
architectures; align the threshold to the in-domain working point first.

Usage
-----
    python training/train.py --image-root /path/to/hand_crops --arch resnet18

See training/README.md for the manifest format, the exact command that
reproduces the shipped weights, and the runtime estimate.
"""
import argparse
import collections
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# The original experiment ran on a Windows console whose code page is GBK, where
# printing any non-ASCII progress text would raise UnicodeEncodeError mid-run.
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

HERE = Path(__file__).resolve().parent

# The manifest lives next to this script, so its default is resolved against the
# script directory and not against the caller's working directory.
DEFAULT_MANIFEST = HERE / "manifest.csv"
DEFAULT_OUT = Path("runs")                   # relative to the caller's cwd

MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet statistics
STD = np.array([0.229, 0.224, 0.225], np.float32)

REQUIRED_COLUMNS = ("label", "hand_id", "hand_tag", "model")

SIZE = 224                                   # overridden by --size

import cv2  # noqa: E402  (imported after the stdout reconfigure above)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def gpu_peak_gb():
    """Peak CUDA allocation so far, in GB.

    Guarded because the original experiment assumed CUDA was always present;
    this port also runs (very slowly) on the CPU fallback above.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def load_sq(p, size=None):
    """Load one crop as a square letterboxed RGB uint8 array.

    The long side is scaled to `size` and the short side is centre-padded with
    black -- not stretched to a square: hand crops are not square and stretching
    them changes the finger-length ratios the classifier is judging. A file that
    cannot be decoded becomes a black square rather than an exception, so one
    corrupt crop cannot kill a multi-hour run.
    """
    size = size or SIZE
    # np.fromfile + imdecode instead of cv2.imread: imread silently returns None
    # for non-ASCII paths on Windows.
    im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        return np.zeros((size, size, 3), np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    s = size / max(h, w)
    im = cv2.resize(im, (max(8, int(w * s)), max(8, int(h * s))),
                    interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    c = np.zeros((size, size, 3), np.uint8)
    c[(size - im.shape[0]) // 2:(size - im.shape[0]) // 2 + im.shape[0],
      (size - im.shape[1]) // 2:(size - im.shape[1]) // 2 + im.shape[1]] = im
    return c


class HandDS(Dataset):
    """One hand crop -> (normalised CHW tensor, 1.0 for good / 0.0 for bad)."""

    def __init__(self, recs, cache, train_mode):
        self.recs, self.cache, self.train_mode = recs, cache, train_mode

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        im = self.cache[r["hand_id"]].copy()
        S = im.shape[0]
        if self.train_mode:
            M = cv2.getRotationMatrix2D((S / 2, S / 2),
                                        float(np.random.uniform(-12, 12)), 1.0)
            im = cv2.warpAffine(im, M, (S, S), borderMode=cv2.BORDER_REPLICATE)
            if np.random.rand() < 0.5:
                k = float(np.random.uniform(0.85, 1.0))
                s0 = int(S * k); o = (S - s0) // 2
                im = cv2.resize(im[o:o + s0, o:o + s0], (S, S),
                                interpolation=cv2.INTER_CUBIC)
            # Colour/background augmentation, aimed at a shortcut that
            # _diag_shortcut2.py measured: inside the Anima group the single
            # scalar "fraction of near-white pixels" alone reached AUC 0.757, and
            # 0.905 inside the "own hands together" gesture -- background
            # brightness on its own predicts the label. A model drifts along that
            # line instead of looking at the hand. The perturbations below
            # specifically destroy colour statistics so the only remaining route
            # is shape.
            if np.random.rand() < 0.15:
                g3 = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                im = cv2.cvtColor(g3, cv2.COLOR_GRAY2RGB)
            if np.random.rand() < 0.6:
                hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-9, 9)) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.6, 1.4), 0, 255)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.8, 1.15), 0, 255)
                im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            if np.random.rand() < 0.5:
                c = float(np.random.uniform(0.75, 1.3))
                im = np.clip((im.astype(np.float32) - 128) * c + 128, 0, 255).astype(np.uint8)
            if np.random.rand() < 0.5:
                im = np.clip(im.astype(np.float32) * float(np.random.uniform(0.9, 1.1))
                             + float(np.random.uniform(-12, 12)), 0, 255).astype(np.uint8)
            # NOTE: there is deliberately no horizontal flip in this block.
            # Left and right hands are anatomically different, and the crop does
            # not carry a side label, so a mirrored sample would be handed to the
            # network under a label that no longer describes it. Every other
            # augmentation above is label-preserving; a flip is not.
        x = (im.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x).permute(2, 0, 1), (1.0 if r["label"] == "good" else 0.0)


def make_model(arch):
    """ImageNet-pretrained backbone with the classification head cut to 1 logit."""
    import torchvision.models as M
    if arch == "resnet18":
        m = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(512, 1)
    elif arch == "resnet34":
        m = M.resnet34(weights=M.ResNet34_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(512, 1)
    elif arch == "efficientnet_b0":
        m = M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(1280, 1)
    elif arch == "mobilenet_v3_small":
        m = M.mobilenet_v3_small(weights=M.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier[3] = nn.Linear(1024, 1)
    else:
        raise ValueError(arch)
    return m


def predict_probs(net, recs, cache, bs):
    """P(good) for every record, with no augmentation and eval mode."""
    dl = DataLoader(HandDS(recs, cache, False), batch_size=bs, shuffle=False)
    ps = []
    with torch.no_grad():
        for xb, _ in dl:
            ps.append(torch.sigmoid(net(xb.to(DEV)).squeeze(-1)).cpu().numpy())
    return np.concatenate(ps) if ps else np.zeros(0)


def load_manifest(manifest_path, image_root):
    """Read the manifest CSV and resolve one absolute image path per row.

    Two manifest shapes are supported:

    * has a ``path`` column -- that column is used as the image location
      (relative values are resolved against the manifest's own directory);
    * no ``path`` column -- crops are addressed as ``<image-root>/<hand_id>.png``,
      which is how the shipped manifest refers to them. ``--image-root`` is then
      mandatory.

    The published manifest has the columns
    ``num,label,hand_id,hand_tag,model,det_conf`` -- ``label`` is ``good`` or
    ``bad``, and there is no path column because the crop images themselves are
    not redistributed with this repository.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as fh:
        # Strip keys and values: hand-edited CSVs grow stray spaces around headers.
        rows = [{k.strip(): (v or "").strip() for k, v in r.items() if k}
                for r in csv.DictReader(fh)]
    if not rows:
        raise SystemExit(f"manifest has no data rows: {manifest_path}")

    columns = set(rows[0])
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing_cols:
        raise SystemExit(
            f"manifest is missing required column(s): {', '.join(missing_cols)}\n"
            f"  found: {', '.join(sorted(columns))}\n"
            f"  required: {', '.join(REQUIRED_COLUMNS)}")

    has_path = "path" in columns
    if not has_path and not image_root:
        raise SystemExit(
            "this manifest has no 'path' column, so crops are addressed as\n"
            "  <image-root>/<hand_id>.png\n"
            "and --image-root is required. The crop images are not redistributed\n"
            "with this repository; point --image-root at your own directory of crops.")

    root = Path(image_root) if image_root else None
    recs, skipped = [], 0
    for r in rows:
        if r.get("label") not in ("good", "bad"):
            skipped += 1
            continue
        if r.get("path"):
            p = Path(r["path"])
            if not p.is_absolute():
                p = manifest_path.parent / p
        elif root is not None:
            p = root / f'{r["hand_id"]}.png'
        else:
            raise SystemExit(
                f"row {r.get('num') or r['hand_id']} has an empty 'path' value and "
                f"--image-root was not given; it needs one or the other")
        recs.append({
            "hand_id": r["hand_id"],
            "label": r["label"],
            "hand_tag": r.get("hand_tag", ""),
            "model": r.get("model", ""),
            "path": str(p),
        })

    if not recs:
        raise SystemExit("manifest contains no rows labelled 'good' or 'bad'")
    if skipped:
        log(f"  skipped {skipped} row(s) whose label is neither 'good' nor 'bad'")

    # Fail fast instead of silently training on black squares: load_sq() maps an
    # undecodable file to zeros, which for a wrong --image-root would mean every
    # sample is black and the run is worthless after hours of compute.
    absent = [r["path"] for r in recs if not Path(r["path"]).is_file()]
    if absent:
        preview = "\n".join(f"  {p}" for p in absent[:5])
        raise SystemExit(
            f"{len(absent)}/{len(recs)} crop images not found. First 5:\n{preview}\n"
            f"Pass --image-root pointing at the directory that holds "
            f"<hand_id>.png files.")
    return recs


def main():
    # Defaults are written into the help strings by hand rather than via
    # ArgumentDefaultsHelpFormatter, so that --help does not print the absolute
    # path of whichever machine built this checkout.
    ap = argparse.ArgumentParser(
        description="Train a small hand-quality CNN (1-logit head, BCE + pos_weight).")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help="CSV with columns num,label,hand_id,hand_tag,model,det_conf, "
                         "where label is 'good' or 'bad'. A 'path' column is used if "
                         "present. (default: manifest.csv next to this script)")
    ap.add_argument("--image-root", default=None,
                    help="directory holding <hand_id>.png crops; required unless the "
                         "manifest already has a 'path' column. The crop images are "
                         "not redistributed with this repo, so you must supply it")
    ap.add_argument("--cross", default=None,
                    help="optional second manifest (same format) drawn from model "
                         "families absent from --manifest, scored after training "
                         "(default: omitted, cross-model evaluation skipped)")
    ap.add_argument("--arch", default="resnet18",
                    choices=["resnet18", "resnet34", "efficientnet_b0",
                             "mobilenet_v3_small"])
    ap.add_argument("--size", type=int, default=224,
                    help="input side length (default: 224). Judging finger structure "
                         "needs detail, and 224 downsamples a ~450px crop, so 320 "
                         "keeps more -- at roughly 2x the training time")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output directory for the .pt/.onnx/oof/cross_pred artifacts "
                         "(default: runs/)")
    ap.add_argument("--epochs", type=int, default=30, help="default: 30")
    ap.add_argument("--bs", type=int, default=32, help="batch size (default: 32)")
    ap.add_argument("--lr", type=float, default=3e-4, help="AdamW lr (default: 3e-4)")
    ap.add_argument("--folds", type=int, default=5, help="default: 5")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed (default: 42). Augmentation draws from the "
                         "*global* numpy RNG, so without fixing this the same data "
                         "differs by ~0.01 OOF AUC between two runs -- which once "
                         "invalidated an ensemble conclusion built on the earlier "
                         "scores")
    ap.add_argument("--select", choices=["best", "last"], default="best",
                    help="best (default) = keep each fold's best-epoch checkpoint, "
                         "chosen on that fold's own validation AUC, before computing "
                         "OOF -- this bakes in optimistic bias; "
                         "last = fixed epoch budget, last epoch, no selection bias, "
                         "the honest estimate")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Fix the randomness up front: augmentation, initialisation and shuffling all
    # draw from it, so the same command reproduces the same scores.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # The resolution is a module-level global read by both load_sq and HandDS, so
    # it has to be set before any image is preloaded.
    global SIZE
    SIZE = args.size

    train = load_manifest(args.manifest, args.image_root)
    log(f"data {len(train)} crops   arch={args.arch}   size={SIZE}   "
        f"select={args.select}   seed={args.seed}   device={DEV}")

    g = sum(1 for r in train if r["label"] == "good")
    log(f"  good {g} / bad {len(train)-g}")
    log(f"  sources {dict(collections.Counter(r['model'] for r in train))}")

    log("preloading images...")
    t0 = time.time()
    cache = {r["hand_id"]: load_sq(r["path"]) for r in train}
    log(f"  done {len(cache)} images  {time.time()-t0:.0f}s")

    from sklearn.model_selection import StratifiedKFold  # noqa: E402
    from sklearn.metrics import roc_auc_score, accuracy_score  # noqa: E402

    # Stratify on "model|hand_tag", not on the label alone. Grouping by source
    # model keeps a whole generation family on one side of the split, and the
    # gesture tag keeps one gesture from being split unevenly, so neither the
    # base model nor the pose can be memorised across folds -- the same key the
    # old CLIP baseline used, which keeps the two sets of numbers comparable.
    strat = np.array([f"{r['model']}|{r['hand_tag']}" for r in train])
    y_all = np.array([1 if r["label"] == "good" else 0 for r in train])
    oof = np.zeros(len(train))
    fold_of = np.zeros(len(train), dtype=int)      # which fold each sample landed in
    fold_best, fold_ep = [], []
    total_t = time.time()
    log(f"class balance {np.bincount(y_all).tolist()}")

    for fold, (tr_i, va_i) in enumerate(
            StratifiedKFold(args.folds, shuffle=True, random_state=42).split(train, strat), 1):
        tr = [train[i] for i in tr_i]
        va = [train[i] for i in va_i]
        net = make_model(args.arch).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        ytr = np.array([1 if r["label"] == "good" else 0 for r in tr])
        # pos_weight = n_negative / n_positive, i.e. up-weight the rare "bad" class
        # inside the loss rather than resampling it.
        pos_w = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=DEV))
        dl_tr = DataLoader(HandDS(tr, cache, True), batch_size=args.bs, shuffle=True,
                           num_workers=0)
        dl_va = DataLoader(HandDS(va, cache, False), batch_size=args.bs, shuffle=False,
                           num_workers=0)
        best_auc, best_state, bad, best_ep = -1.0, None, 0, 0
        for ep in range(args.epochs):
            net.train()
            for xb, yb in dl_tr:
                xb, yb = xb.to(DEV), yb.to(DEV)
                opt.zero_grad()
                lossf(net(xb).squeeze(-1), yb).backward()
                opt.step()
            sched.step()
            net.eval()
            ps, ys = [], []
            with torch.no_grad():
                for xb, yb in dl_va:
                    ps.append(torch.sigmoid(net(xb.to(DEV)).squeeze(-1)).cpu().numpy())
                    ys.append(yb.numpy())
            p, yy = np.concatenate(ps), np.concatenate(ys)
            a = roc_auc_score(yy, p) if len(set(yy)) > 1 else 0.5
            if a > best_auc:
                best_auc, bad, best_ep = a, 0, ep + 1
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                # Early stopping: 8 consecutive epochs without a new best. Skipped
                # for select=last, whose whole point is to run the fixed budget out.
                if bad >= 8 and args.select == "best":
                    break
            if ep % 5 == 0 or ep == args.epochs - 1:
                log(f"  fold {fold} ep {ep+1}/{args.epochs} AUC={a:.4f} "
                    f"best={best_auc:.4f} peak={gpu_peak_gb():.2f}GB")
        # Careful: loading best_state means choosing the checkpoint on *this fold's
        # own validation score* and then using it for the OOF prediction, so this
        # OOF AUC carries optimistic bias (flagged by an independent audit).
        # --select last exists to quantify that bias.
        if args.select == "best" and best_state:
            net.load_state_dict(best_state)
        net.eval()
        fold_of[va_i] = fold
        oof[va_i] = predict_probs(net, va, cache, args.bs)
        fold_best.append(best_auc)
        fold_ep.append(best_ep)
        log(f"  fold {fold} done  best={best_auc:.4f} @ep{best_ep}")
        del net
        torch.cuda.empty_cache()

    auc = roc_auc_score(y_all, oof)
    log(f"=== {args.arch} OOF AUC = {auc:.4f}  "
        f"accuracy {accuracy_score(y_all, oof>=0.5):.4f} ===")
    log(f"    per fold {[round(x,3) for x in fold_best]}   spread {np.std(fold_best):.3f}")

    tags = np.array([r["hand_tag"] for r in train])
    models = np.array([r["model"] for r in train])
    inner = []
    log("=== by gesture ===")
    for t in sorted(set(tags)):
        m = tags == t
        yy = y_all[m]
        # Skip cells too small to carry an AUC at all.
        if m.sum() < 8 or yy.sum() < 3 or (1 - yy).sum() < 3:
            continue
        a = roc_auc_score(yy, oof[m]); inner.append(a)
        log(f"  {t:<28} n={m.sum():>3} (g{int(yy.sum())}/b{int((1-yy).sum())}) AUC={a:.3f}")
    if inner:
        log(f"    mean within-gesture {np.mean(inner):.4f}")
    for mdl in sorted(set(models)):
        m = models == mdl
        if y_all[m].sum() < 3 or (1 - y_all[m]).sum() < 3:
            continue
        log(f"  by model {mdl:<14} AUC={roc_auc_score(y_all[m], oof[m]):.3f}")

    # -- retrain on all data and save the final model --
    log("retraining on the full set and saving the final model")
    net = make_model(args.arch).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    pos_w = float((y_all == 0).sum()) / max(float((y_all == 1).sum()), 1.0)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=DEV))
    dl = DataLoader(HandDS(train, cache, True), batch_size=args.bs, shuffle=True)
    # The full set has no validation split, so use the mean of the per-fold best
    # epochs as the budget. (An earlier version wrote
    # np.mean([len(fold_best)])*3, where len() is always the fold count, so it
    # silently trained 12 epochs every time -- a typo, now fixed.)
    epochs_final = max(8, int(round(np.mean(fold_ep))))
    log(f"  full-data run: {epochs_final} epochs (per-fold best epochs {fold_ep})")
    for ep in range(epochs_final):
        net.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            lossf(net(xb).squeeze(-1), yb).backward()
            opt.step()
    net.eval()
    # The filename carries the resolution and the selection mode so that
    # experiments cannot overwrite each other: one rerun once reused the same
    # filename, clobbered the previous OOF scores, and invalidated an ensemble
    # conclusion that had already been written up.
    tag = args.arch if SIZE == 224 else f"{args.arch}_s{SIZE}"
    if args.select != "best":
        tag += f"_{args.select}"
    sd = out / f"{tag}_handqc.pt"
    torch.save(net.state_dict(), str(sd))
    n_par = sum(p.numel() for p in net.parameters())
    log(f"[out] {sd}   {n_par/1e6:.1f}M params  {sd.stat().st_size/1e6:.1f} MB")

    # Export ONNX so the model can be deployed without PyTorch.
    # Note: torch 2.9's torch.onnx.export defaults to dynamo=True, and that path
    # requires onnxscript -- without it the whole export raises
    # "No module named 'onnxscript'". The legacy tracer (dynamo=False) needs no
    # extra dependency, hence the explicit flag.
    try:
        onx = out / f"{tag}_handqc.onnx"
        dummy = torch.randn(1, 3, SIZE, SIZE, device=DEV)
        torch.onnx.export(net, dummy, str(onx), input_names=["input"],
                          output_names=["logit"], dynamic_axes={"input": {0: "batch"}},
                          opset_version=14, dynamo=False)
        log(f"[out] {onx}   {onx.stat().st_size/1e6:.1f} MB")
    except Exception as e:
        log(f"ONNX export failed: {type(e).__name__} {e}")

    np.savez_compressed(str(out / f"oof_{tag}.npz"), oof=oof, y=y_all,
                        fold=fold_of, seed=args.seed, select=args.select)

    # -- cross-model test: model families that never appeared during training --
    if args.cross and Path(args.cross).exists():
        cross = load_manifest(args.cross, args.image_root)
        if len(cross) >= 10:
            log("=" * 60)
            log(f"cross-model set: {len(cross)} crops from families absent in training")
            log(f"  sources {dict(collections.Counter(r['model'] for r in cross))}")
            ccache = {r["hand_id"]: load_sq(r["path"]) for r in cross}
            pc = predict_probs(net, cross, ccache, args.bs)
            yc = np.array([1 if r["label"] == "good" else 0 for r in cross])
            gc = int(yc.sum())
            nb = len(yc) - gc
            log(f"  good {gc} / bad {nb}")

            if gc > 0 and nb > 0:
                log(f"  cross-model AUC = {roc_auc_score(yc, pc):.4f}")
            else:
                # Newer models (FLUX.2, Krea2) rarely draw a wrong hand at all, so
                # the set can end up with zero bad samples and no AUC exists. The
                # metric that still matters is answerable: on an unseen art style,
                # does the model flag good hands as bad? A model that memorised
                # IL style fingerprints would false-positive massively here.
                log("  [note] no bad samples in this set, so AUC is undefined.")
                log("         Falling back to the false-positive rate -- how often a")
                log("         good hand is called bad on an unseen style, which is the")
                log("         most direct evidence of generalisation available.")
            if gc > 0:
                gs = pc[yc == 1]
                for thr in (0.3, 0.5, 0.7):
                    fa = float((gs < thr).mean())
                    log(f"    threshold {thr:.1f}: false-positive rate {fa*100:5.1f}%  "
                        f"({int((gs<thr).sum())}/{gc})")
                log(f"    good-hand score median {np.median(gs):.3f}  "
                    f"mean {gs.mean():.3f}  "
                    f"p10 {np.percentile(gs,10):.3f}  "
                    f"p90 {np.percentile(gs,90):.3f}")
            if nb > 0:
                bs_ = pc[yc == 0]
                log(f"    bad-hand score median {np.median(bs_):.3f}  mean {bs_.mean():.3f}")
            for mdl in sorted(set(r["model"] for r in cross)):
                m = np.array([r["model"] == mdl for r in cross])
                if 0 < yc[m].sum() < m.sum():
                    log(f"    by model {mdl:<8} n={m.sum():>3} "
                        f"(g{int(yc[m].sum())}/b{int((1-yc[m]).sum())}) "
                        f"AUC={roc_auc_score(yc[m], pc[m]):.3f}")
                elif yc[m].sum() == m.sum():
                    gs = pc[m]
                    log(f"    by model {mdl:<8} n={m.sum():>3} all good, "
                        f"FPR@0.5={float((gs<0.5).mean())*100:.1f}%  "
                        f"median score {np.median(gs):.3f}")
            tags_c = np.array([r["hand_tag"] for r in cross])
            log("    by gesture:")
            for t in sorted(set(tags_c)):
                m = tags_c == t
                if m.sum() < 6:
                    continue
                if yc[m].sum() > 0 and (1 - yc[m]).sum() > 0:
                    log(f"      {t:<28} n={m.sum():>3} "
                        f"(g{int(yc[m].sum())}/b{int((1-yc[m]).sum())}) "
                        f"AUC={roc_auc_score(yc[m], pc[m]):.3f}")
                else:
                    gs = pc[m]
                    log(f"      {t:<28} n={m.sum():>3} all good  "
                        f"FPR@0.5={float((gs<0.5).mean())*100:5.1f}%  "
                        f"median {np.median(gs):.3f}")
            (out / f"cross_pred_{tag}.json").write_text(json.dumps(
                [{"hand_id": r["hand_id"], "model": r["model"],
                  "hand_tag": r["hand_tag"], "label": r["label"],
                  "score": float(s)} for r, s in zip(cross, pc)],
                ensure_ascii=False, indent=1), encoding="utf-8")
            log(f"  [out] {out / f'cross_pred_{tag}.json'}")

    log(f"total wall clock {(time.time()-total_t)/60:.1f} min")


if __name__ == "__main__":
    main()
