"""Training pipeline for the robust 20-class image classifier.

Run from anywhere (paths are resolved relative to this file, not the cwd):

    python submissions/my_team/train.py              # full training run
    python submissions/my_team/train.py --smoke      # fast end-to-end check
    python submissions/my_team/train.py --epochs 60 --batch-size 128

Output: weights.joblib (a CPU state_dict) written next to this file, exactly
as the grader expects.

Robustness uses an augmentation curriculum: train with light geometric aug
until clean-val passes a threshold, then switch on heavy colour/occlusion aug
(`build_light_transform` -> `build_heavy_transform`). Rotation is deliberately
kept out of training so `random_rotation` stays a valid frozen OOD-test.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import make_grid

# --- Make the project-root modules (base_model, labels) importable no matter
#     where this script is launched from. ----------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base_model import ImageNetSubset  # noqa: E402
from labels import IDX_TO_NAME  # noqa: E402
from model import ModelArchitecture  # noqa: E402

_MPL_OK = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _MPL_OK = True
except ImportError:
    pass

# --- Paths ------------------------------------------------------------------
DATA_ROOT = PROJECT_ROOT / "dataset"
AUG_DIR = DATA_ROOT / "augmentations"
OUTPUT = HERE / "weights.joblib"
TB_LOGDIR = HERE / "runs"

# --- Fixed constants --------------------------------------------------------
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 20
SEED = 42
VAL_FRACTION = 0.1

# OOD policy: of the two provided augmentation types, one is the *tuning* signal
# (may influence model selection) and the other is a frozen *test* that we look
# at exactly once, on the final selected checkpoint. This simulates the grader's
# real setup -- it scores us on manipulations we never got to tune against.
OOD_VAL_NAME = "color_jitter"      # tuning signal -> enters the selection score
OOD_TEST_NAME = "random_rotation"  # frozen -> evaluated only at the very end


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def build_light_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Phase-1 (pre-threshold) augmentation: gentle geometry only.

    Heavy colour/occlusion augmentation from step 0 slows a from-scratch model.
    Phase 1 keeps it near-clean so the network can build solid features fast.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_heavy_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Phase-2 (post-threshold) augmentation: the robustness driver.

    Switched on once the model passes a clean-val threshold. Targets the exact
    manipulations in the brief -- colour, lighting, occlusion, framing.

    NOTE: deliberately NO rotation. `random_rotation` is our *frozen* OOD-test;
    training on rotation would turn it in-distribution and void the honest test.
    Colour jitter IS included -- `color_jitter` is the tuning OOD signal.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ])


def build_eval_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Must match the grader's preprocessing exactly (see evaluate.py).

    `image_size` is only overridden for the smoke test; at the default it
    reproduces the grader's 256-resize / 224-crop exactly.
    """
    resize = round(image_size * 256 / IMAGE_SIZE)
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------
def collect_aug_originals() -> set:
    """Filenames of train images that have a provided augmented counterpart.

    `augmentations/<type>/<class>/00_goldfish_000043_<type>.jpg` was generated
    from `train/<class>/00_goldfish_000043.jpg`. We strip the "_<type>" suffix
    to recover the original filename so those originals can be kept OUT of the
    training set -- otherwise the OOD probe leaks (the model would have seen the
    clean version of every image it is being stress-tested on).
    """
    names = set()
    if not AUG_DIR.exists():
        return names
    for p in AUG_DIR.rglob("*.jpg"):
        aug_type = p.parent.parent.name  # e.g. "color_jitter"
        stem = p.stem
        suffix = "_" + aug_type
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
        names.add(stem + ".jpg")
    return names


def make_splits():
    """Leakage-safe train/val *index* split over dataset/train.

    Returns indices (not Subsets) so the train transform can be swapped mid-run
    for the augmentation curriculum without re-deciding the split.
    """
    samples = ImageNetSubset(DATA_ROOT, "train").samples
    n = len(samples)

    aug_originals = collect_aug_originals()
    forced_val = [i for i, (p, _) in enumerate(samples) if p.name in aug_originals]
    forced = set(forced_val)

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n, generator=g).tolist()
    rest = [i for i in perm if i not in forced]

    target_val = int(n * VAL_FRACTION)
    extra = max(0, target_val - len(forced_val))
    val_idx = forced_val + rest[:extra]
    train_idx = rest[extra:]

    print(
        f"Split: {len(train_idx)} train / {len(val_idx)} val "
        f"({len(forced_val)} val images held out to protect the OOD probe)"
    )
    return train_idx, val_idx


def build_train_loader(dataset, batch_size, workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=workers, drop_last=True,
                      persistent_workers=workers > 0)


def build_ood_loaders(eval_tf, batch_size, workers):
    """One DataLoader per provided augmentation type (local OOD stress test)."""
    loaders = {}
    if not AUG_DIR.exists():
        return loaders
    for sub in sorted(d.name for d in AUG_DIR.iterdir() if d.is_dir()):
        try:
            ds = ImageNetSubset(AUG_DIR, sub, transform=eval_tf)
        except (FileNotFoundError, RuntimeError):
            continue  # not a complete class-labelled folder (e.g. a stray dir)
        loaders[sub] = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                  num_workers=workers)
    return loaders


# ---------------------------------------------------------------------------
# Train / eval steps
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    # Keep running stats on-device and sync only once per epoch. Calling
    # .item() every step forces a host<->GPU sync that serializes MPS and
    # tanks throughput (~5x slower in practice).
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    total = 0
    for x, y in loader:
        # NB: do NOT use non_blocking=True here. On MPS the async copy can race
        # with the next kernel, feeding *garbage labels* into the loss -- which
        # silently corrupts training (model collapses) and inflates accuracy.
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_sum += loss.detach() * y.size(0)
        correct += (logits.argmax(1) == y).sum()
        total += y.size(0)
    return (loss_sum / total).item(), (correct / total).item()


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = torch.zeros((), device=device)
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum()
        total += y.size(0)
    return (correct / total).item() if total else 0.0


def build_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs=3):
    warmup = warmup_epochs * steps_per_epoch
    total = epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# TensorBoard helpers
# ---------------------------------------------------------------------------
def log_weight_histograms(model, writer, step):
    if writer is None:
        return
    for name, param in model.named_parameters():
        writer.add_histogram(f"weights/{name}", param.detach().cpu(), step)
        if param.grad is not None:
            writer.add_histogram(f"grads/{name}", param.grad.detach().cpu(), step)


def log_sample_images(loader, writer, tag, n=32):
    if writer is None:
        return
    images, _ = next(iter(loader))
    images = images[:n]
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    images = (images * std + mean).clamp(0, 1)
    writer.add_image(tag, make_grid(images, nrow=8), 0)


def log_confusion_matrix(model, loader, device, writer, class_names, step, tag):
    if not _MPL_OK or writer is None:
        return
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1).cpu().numpy()
            for t, p in zip(y.cpu().numpy(), preds):
                cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    writer.add_figure(tag, fig, step)
    plt.close(fig)


def log_predictions_grid(model, loader, device, writer, class_names, step, tag):
    if not _MPL_OK or writer is None:
        return
    images, labels = next(iter(loader))
    n = min(16, len(images))
    images, labels = images[:n], labels[:n]
    model.eval()
    with torch.no_grad():
        preds = model(images.to(device)).argmax(1).cpu()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    imgs_show = (images * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    for i in range(n):
        correct = preds[i].item() == labels[i].item()
        axes[i].imshow(imgs_show[i])
        axes[i].set_title(
            f"P: {class_names[preds[i]]}\nT: {class_names[labels[i]]}",
            color="green" if correct else "red", fontsize=7)
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.tight_layout()
    writer.add_figure(tag, fig, step)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train the robust 20-class classifier.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--aug-threshold", type=float, default=0.6,
                   help="Clean-val accuracy that triggers heavy augmentation.")
    p.add_argument("--force-switch-frac", type=float, default=0.4,
                   help="Force the heavy-aug switch by this fraction of epochs "
                        "even if the threshold was never reached.")
    p.add_argument("--smoke", action="store_true",
                   help="Fast end-to-end check: 2 epochs on a small data subset.")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    # Smoke runs at a small resolution on a tiny subset: the point is to
    # exercise the *pipeline* (loading, curriculum switch, TB logging,
    # checkpoint save/reload) fast -- not to reach real accuracy. 224x224 on
    # CPU is minutes/epoch; 64x64 is seconds/epoch so the board fills quickly.
    img_size = 64 if args.smoke else IMAGE_SIZE

    eval_tf = build_eval_transform(img_size)
    train_idx, val_idx = make_splits()
    # `train_full` carries the *current* augmentation; we mutate its .transform
    # at the curriculum switch and rebuild the loader so workers pick it up.
    train_full = ImageNetSubset(DATA_ROOT, "train", transform=build_light_transform(img_size))
    val_full = ImageNetSubset(DATA_ROOT, "train", transform=eval_tf)
    train_set = Subset(train_full, train_idx)
    val_set = Subset(val_full, val_idx)

    # Val/OOD sets are small; single-process decode (~389 img/s) already
    # outpaces eval compute, so 0 workers avoids needless per-epoch respawning.
    ood_loaders = build_ood_loaders(eval_tf, args.batch_size, workers=0)
    ood_val_loader = ood_loaders.get(OOD_VAL_NAME)    # tuning signal
    ood_test_loader = ood_loaders.get(OOD_TEST_NAME)  # frozen until the end

    epochs = args.epochs
    if args.smoke:
        epochs = 2
        # Sample randomly (not range()) so the subsets stay class-balanced --
        # the splits are class-ordered, so range() would grab one class.
        g = torch.Generator().manual_seed(SEED)
        ti = torch.randperm(len(train_set), generator=g)[:256].tolist()
        vi = torch.randperm(len(val_set), generator=g)[:128].tolist()
        train_set, val_set = Subset(train_set, ti), Subset(val_set, vi)
        print(f"[smoke] {len(train_set)} train / {len(val_set)} val, "
              f"{epochs} epochs, {img_size}x{img_size} images")

    train_loader = build_train_loader(train_set, args.batch_size, args.workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    model = ModelArchitecture(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay, nesterov=True)
    scheduler = build_scheduler(optimizer, epochs, len(train_loader))

    total_params = sum(p.numel() for p in model.parameters())
    class_names_list = [IDX_TO_NAME[i] for i in range(NUM_CLASSES)]

    # TensorBoard logging (optional). Each run gets an isolated timestamped dir.
    #   tensorboard --logdir submissions/my_team/runs
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        ts = time.strftime("%Y%m%d_%H%M%S")
        _arch_name = type(model).__name__.lower()
        run_dir = TB_LOGDIR / f"{_arch_name}_{ts}"
        writer = SummaryWriter(log_dir=str(run_dir))
        print(f"TensorBoard: tensorboard --logdir {TB_LOGDIR}  (run: {run_dir.name})")

        try:
            _sample = torch.zeros(1, 3, img_size, img_size, device=device)
            writer.add_graph(model, _sample)
        except Exception as _e:
            print(f"add_graph skipped: {_e}")

        writer.add_text("model/architecture",
            f"**Class**: `{type(model).__name__}`\n\n"
            f"**Total params**: {total_params:,}\n\n"
            f"```\n{model}\n```", 0)

        _n_ood_val = len(ood_val_loader.dataset) if ood_val_loader else "N/A"
        _n_ood_test = len(ood_test_loader.dataset) if ood_test_loader else "N/A"
        writer.add_text("data/info", "\n".join([
            "| Key | Value |", "| --- | --- |",
            f"| dataset_root | `{DATA_ROOT}` |",
            f"| classes | {', '.join(class_names_list)} |",
            f"| num_train | {len(train_set)} |",
            f"| num_val | {len(val_set)} |",
            f"| ood_val ({OOD_VAL_NAME}) | {_n_ood_val} |",
            f"| ood_test ({OOD_TEST_NAME}) | {_n_ood_test} |",
            f"| split_seed | {SEED} |",
            f"| aug_threshold | {args.aug_threshold} |",
            f"| force_switch_frac | {args.force_switch_frac} |",
            f"| phase1 | RandomResizedCrop(0.7-1.0) + HFlip |",
            f"| phase2 | adds ColorJitter + RandomGrayscale + RandomErasing |",
        ]), 0)

        log_sample_images(train_loader, writer, tag="data/sample_images")

    except Exception as e:  # pragma: no cover - logging must never break training
        print(f"(TensorBoard disabled: {e})")

    best_score = -1.0
    best_state = None
    best = {}

    # Curriculum state: phase 1 = light aug, phase 2 = heavy aug. We switch when
    # clean-val crosses the threshold, or force it by force_switch_epoch so a
    # model that plateaus below the bar still gets its robustness training.
    phase = 1
    force_switch_epoch = max(1, int(epochs * args.force_switch_frac))
    print(f"Curriculum: light aug -> heavy at clean-val>={args.aug_threshold:.2f} "
          f"or epoch {force_switch_epoch} (whichever first)")

    for epoch in range(1, epochs + 1):
        _epoch_start = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_acc = evaluate(model, val_loader, device)
        # OOD-val (color_jitter) is allowed to influence selection; OOD-test
        # (random_rotation) is deliberately NOT evaluated here -- it stays frozen.
        ood_val_acc = evaluate(model, ood_val_loader, device) if ood_val_loader else val_acc
        # Selection metric mirrors the 50/50 clean-vs-OOD grading.
        score = 0.5 * val_acc + 0.5 * ood_val_acc

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:3d}/{epochs} | lr {lr_now:.4f} | phase {phase} | "
            f"train_loss {tr_loss:.3f} acc {tr_acc:.3f} | "
            f"val {val_acc:.3f} | ood_val {ood_val_acc:.3f} | score {score:.3f}"
        )

        if writer is not None:
            writer.add_scalar("loss/train", tr_loss, epoch)
            writer.add_scalar("acc/train", tr_acc, epoch)
            writer.add_scalar("acc/val_clean", val_acc, epoch)
            writer.add_scalar("acc/ood_val", ood_val_acc, epoch)
            writer.add_scalar("acc/score", score, epoch)
            writer.add_scalar("lr", lr_now, epoch)
            writer.add_scalar("curriculum/phase", phase, epoch)
            writer.add_scalar("time/epoch_sec", time.time() - _epoch_start, epoch)
            if epoch % 5 == 0:
                log_weight_histograms(model, writer, epoch)

        if score > best_score:
            best_score = score
            best = {"epoch": epoch, "val": val_acc, "ood_val": ood_val_acc}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Curriculum switch (after this epoch's eval, so it affects the next one).
        if phase == 1 and (val_acc >= args.aug_threshold or epoch >= force_switch_epoch):
            reason = "val>=thresh" if val_acc >= args.aug_threshold else "force-by-epoch"
            train_full.transform = build_heavy_transform(img_size)
            train_loader = build_train_loader(train_set, args.batch_size, args.workers)
            phase = 2
            print(f"  -> [curriculum] HEAVY augmentation ON at epoch {epoch} "
                  f"({reason}, val={val_acc:.3f})")

    # Restore the selected checkpoint, then unfreeze the OOD-test ONCE on it.
    assert best_state is not None
    model.load_state_dict(best_state)
    ood_test_acc = evaluate(model, ood_test_loader, device) if ood_test_loader else float("nan")
    honest_score = 0.5 * best["val"] + 0.5 * ood_test_acc

    print(
        f"\nSelected epoch {best['epoch']}: "
        f"clean_val {best['val']:.3f} | ood_val ({OOD_VAL_NAME}) {best['ood_val']:.3f}"
    )
    print(
        f"FROZEN OOD-test ({OOD_TEST_NAME}): {ood_test_acc:.3f}  "
        f"-> honest 50/50 estimate {honest_score:.3f}"
    )
    if writer is not None:
        writer.add_scalar("acc/ood_test_frozen", ood_test_acc, best["epoch"])

        log_confusion_matrix(model, val_loader, device, writer,
                             class_names_list, step=best["epoch"],
                             tag="eval/confusion_matrix")
        log_predictions_grid(model, val_loader, device, writer,
                             class_names_list, step=best["epoch"],
                             tag="eval/predictions")
        log_weight_histograms(model, writer, step=best["epoch"])

        hparams = {
            "epochs":            args.epochs,
            "batch_size":        args.batch_size,
            "lr":                args.lr,
            "weight_decay":      args.weight_decay,
            "label_smoothing":   args.label_smoothing,
            "aug_threshold":     args.aug_threshold,
            "force_switch_frac": args.force_switch_frac,
            "total_params":      int(total_params),
        }
        metrics = {
            "hparam/val_clean":       best["val"],
            "hparam/ood_val":         best["ood_val"],
            "hparam/ood_test_frozen": ood_test_acc,
            "hparam/honest_score":    honest_score,
        }
        writer.add_hparams(hparams, metrics)

        rows = [f"| {k} | {v} |" for k, v in {**hparams, **metrics}.items()]
        writer.add_text("run/summary",
            "| Param | Value |\n| --- | --- |\n" + "\n".join(rows), 0)

        writer.flush()
        writer.close()

    # Persist the best checkpoint as a CPU state_dict (hardware-independent).
    joblib.dump(best_state, OUTPUT)
    print(f"Saved best weights to {OUTPUT}")

    # Sanity check: the saved file must reload into a fresh architecture,
    # exactly as the grader's predict.py does.
    reloaded = ModelArchitecture(num_classes=NUM_CLASSES)
    reloaded.load_state_dict(joblib.load(OUTPUT))
    print("Reload check OK: weights.joblib matches ModelArchitecture.")


if __name__ == "__main__":
    main()
