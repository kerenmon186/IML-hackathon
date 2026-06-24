"""Training pipeline for the robust 20-class image classifier.

Run from anywhere (paths are resolved relative to this file, not the cwd):

    python submissions/my_team/train.py              # full training run
    python submissions/my_team/train.py --smoke      # fast end-to-end check
    python submissions/my_team/train.py --epochs 60 --batch-size 128

Output: weights.joblib (a CPU state_dict) written next to this file, exactly
as the grader expects.

This is the *baseline* pipeline. It already bakes in colour/lighting/geometry
robustness augmentations (the cheap, high-value ones). The heavier robustness
ideas -- background replacement, style/texture randomization, etc. -- plug in
at `build_train_transform()` and are the focus of the next iteration.
"""

import argparse
import math
import sys
from pathlib import Path

import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# --- Make the project-root modules (base_model, labels) importable no matter
#     where this script is launched from. ----------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base_model import ImageNetSubset  # noqa: E402
from model import ModelArchitecture  # noqa: E402

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
def build_train_transform() -> transforms.Compose:
    """Training-time augmentation.

    The eval transform below is fixed by the grader; everything that makes the
    model *robust* lives here. The current set targets exactly the kinds of
    held-out manipulations described in the brief:
      - ColorJitter / RandomGrayscale  -> colour & lighting distortions
      - RandomResizedCrop / flip       -> scale, position, framing
      - RandomErasing                  -> occlusion / missing context
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ])


def build_eval_transform() -> transforms.Compose:
    """Must match the grader's preprocessing exactly (see evaluate.py)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
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


def make_splits(train_tf, eval_tf):
    """Build leakage-safe train / validation Subsets over dataset/train."""
    train_full = ImageNetSubset(DATA_ROOT, "train", transform=train_tf)
    eval_full = ImageNetSubset(DATA_ROOT, "train", transform=eval_tf)
    samples = train_full.samples
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

    train_set = Subset(train_full, train_idx)
    val_set = Subset(eval_full, val_idx)
    print(
        f"Split: {len(train_idx)} train / {len(val_idx)} val "
        f"({len(forced_val)} val images held out to protect the OOD probe)"
    )
    return train_set, val_set


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
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total if total else 0.0


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
    p.add_argument("--smoke", action="store_true",
                   help="Fast end-to-end check: 2 epochs on a small data subset.")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    train_tf, eval_tf = build_train_transform(), build_eval_transform()
    train_set, val_set = make_splits(train_tf, eval_tf)
    ood_loaders = build_ood_loaders(eval_tf, args.batch_size, args.workers)

    epochs = args.epochs
    if args.smoke:
        epochs = 2
        train_set = Subset(train_set, list(range(min(len(train_set), 1500))))
        val_set = Subset(val_set, list(range(min(len(val_set), 500))))
        print(f"[smoke] {len(train_set)} train / {len(val_set)} val, {epochs} epochs")

    persistent = args.workers > 0
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True,
                              persistent_workers=persistent)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, persistent_workers=persistent)

    model = ModelArchitecture(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay, nesterov=True)
    scheduler = build_scheduler(optimizer, epochs, len(train_loader))

    # TensorBoard logging (optional, available in this env). Launch with:
    #   tensorboard --logdir submissions/my_team/runs
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(TB_LOGDIR))
    except Exception as e:  # pragma: no cover - logging must never break training
        print(f"(TensorBoard disabled: {e})")

    best_score = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_acc = evaluate(model, val_loader, device)
        ood_accs = {name: evaluate(model, ld, device) for name, ld in ood_loaders.items()}
        mean_ood = sum(ood_accs.values()) / len(ood_accs) if ood_accs else val_acc
        # Selection metric mirrors the 50/50 clean-vs-OOD grading.
        score = 0.5 * val_acc + 0.5 * mean_ood

        lr_now = optimizer.param_groups[0]["lr"]
        ood_str = " ".join(f"{k}={v:.3f}" for k, v in ood_accs.items())
        print(
            f"epoch {epoch:3d}/{epochs} | lr {lr_now:.4f} | "
            f"train_loss {tr_loss:.3f} acc {tr_acc:.3f} | "
            f"val {val_acc:.3f} | ood[{ood_str}] mean {mean_ood:.3f} | score {score:.3f}"
        )

        if writer is not None:
            writer.add_scalar("loss/train", tr_loss, epoch)
            writer.add_scalar("acc/train", tr_acc, epoch)
            writer.add_scalar("acc/val_clean", val_acc, epoch)
            writer.add_scalar("acc/ood_mean", mean_ood, epoch)
            writer.add_scalar("acc/score", score, epoch)
            for k, v in ood_accs.items():
                writer.add_scalar(f"acc/ood_{k}", v, epoch)
            writer.add_scalar("lr", lr_now, epoch)

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if writer is not None:
        writer.close()

    # Persist the best checkpoint as a CPU state_dict (hardware-independent).
    assert best_state is not None
    joblib.dump(best_state, OUTPUT)
    print(f"Saved best weights (score={best_score:.3f}) to {OUTPUT}")

    # Sanity check: the saved file must reload into a fresh architecture,
    # exactly as the grader's predict.py does.
    reloaded = ModelArchitecture(num_classes=NUM_CLASSES)
    reloaded.load_state_dict(joblib.load(OUTPUT))
    print("Reload check OK: weights.joblib matches ModelArchitecture.")


if __name__ == "__main__":
    main()
