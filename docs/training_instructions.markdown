# Training Instructions

## Quick Start

```bash
# Full training run (default hyperparameters)
python submissions/my_team/train.py

# Fast end-to-end pipeline check (2 epochs, 64x64, 256/128-image subset)
python submissions/my_team/train.py --smoke

# Custom run
python submissions/my_team/train.py --epochs 60 --batch-size 128 --lr 0.05
```

## Hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `50` | Total training epochs |
| `--batch-size` | `64` | Mini-batch size |
| `--lr` | `0.1` | Initial learning rate (cosine-decayed after warmup) |
| `--weight-decay` | `5e-4` | L2 regularization strength |
| `--workers` | `4` | DataLoader worker processes |
| `--label-smoothing` | `0.1` | CrossEntropyLoss label smoothing factor |
| `--aug-threshold` | `0.6` | Clean-val accuracy that triggers the switch to heavy augmentation |
| `--force-switch-frac` | `0.4` | Force the heavy-aug switch by this fraction of total epochs even if the threshold was never reached |
| `--smoke` | `False` | Run a 2-epoch smoke test on a tiny subset |

## Learning Rate Schedule

Linear warmup for 3 epochs, then cosine annealing to 0 over the remaining epochs. The schedule is applied per-step (not per-epoch) so it's invariant to batch size.

## Augmentation Curriculum

Training runs in two phases:

- **Phase 1 (light aug):** `RandomResizedCrop` + `RandomHorizontalFlip` only. Lets the model build solid features quickly from scratch.
- **Phase 2 (heavy aug):** Adds `ColorJitter`, `RandomGrayscale`, and `RandomErasing`. Switches on when clean-val accuracy exceeds `--aug-threshold`, or is forced at `--force-switch-frac` of total epochs, whichever comes first.

No rotation augmentation is used — `random_rotation` is reserved as the frozen OOD test.

## Smoke Test

`--smoke` exercises the *whole pipeline* (data loading, the curriculum switch, TensorBoard logging, checkpoint save + reload) in seconds rather than minutes — it is **not** meant to reach real accuracy:

- 2 epochs
- 64×64 images (vs 224×224 in a real run) — the eval transform's resize scales with image size, so the 256/224 ratio is preserved
- A class-balanced 256-train / 128-val random subset

## Outputs

- **`submissions/my_team/weights.joblib`** — best checkpoint (CPU state dict), selected by the composite score `0.5 * clean_val + 0.5 * ood_val (color_jitter)`. The script reloads it into a fresh `ModelArchitecture` at the end as a grader-compatibility sanity check.
- **`submissions/my_team/runs/<arch>_<timestamp>/`** — TensorBoard logs. **Each run writes to its own timestamped directory** (e.g. `modelarchitecture_20260625_143000/`), so successive runs accumulate side-by-side for comparison rather than overwriting one another.

```bash
# View TensorBoard (point at the parent dir to see all runs at once)
tensorboard --logdir submissions/my_team/runs
```

### What gets logged

| TensorBoard tab | Tag(s) | Content |
|-----------------|--------|---------|
| SCALARS | `loss/train`, `loss/val_clean`, `loss/ood_val`, `acc/train`, `acc/val_clean`, `acc/ood_val`, `acc/score`, `lr`, `curriculum/phase`, `time/epoch_sec` | Per-epoch training/eval curves (train + **validation loss** on both clean and color_jitter splits) |
| SCALARS | `acc/ood_test_frozen`, `loss/ood_test_frozen` | Frozen `random_rotation` accuracy + loss (logged once, at the selected epoch) |
| GRAPHS | — | The model compute graph |
| IMAGES | `data/sample_images` | A grid of (de-normalized) training images |
| IMAGES | `eval/confusion_matrix`, `eval/predictions` | Final confusion matrix + a labeled prediction grid (green=correct, red=wrong) |
| DISTRIBUTIONS / HISTOGRAMS | `weights/*`, `grads/*` | Per-parameter weight & gradient histograms (every 5 epochs + at the end) |
| TEXT | `model/architecture`, `data/info`, `run/summary` | Model summary, dataset/split info, final hparams + metrics table |
| HPARAMS | `hparam/val_clean`, `hparam/ood_val`, `hparam/ood_test_frozen`, `hparam/honest_score` | Hyperparameters vs final metrics, for cross-run comparison |

The confusion matrix and prediction grid require **matplotlib** (and numpy); if they are not installed those two figures are silently skipped and the rest of the logging still runs. All TensorBoard logging is best-effort and never breaks training.

### Loss tracking

`evaluate()` returns `(acc, loss)` when passed the training criterion, so the
loop now logs **validation loss** (CrossEntropy with the same `label_smoothing`
as training) alongside accuracy. The per-epoch stdout line carries the clean-val
loss, e.g.:

```
epoch  45/50 | lr 0.0028 | phase 2 | train_loss 0.696 acc 0.971 | val 0.924 loss 0.812 | ood_val 0.940 | score 0.932
```

and `loss/val_clean`, `loss/ood_val`, `loss/ood_test_frozen` are written to
TensorBoard for the loss curves. Watching val loss (not just val accuracy) is
the cleaner overfitting signal: accuracy can plateau while loss starts creeping
back up. Calling `evaluate()` *without* a criterion still returns a bare float,
so older call sites are unaffected.

## Post-run Evaluation (`evaluate_model.py`)

[`submissions/my_team/evaluate_model.py`](../submissions/my_team/evaluate_model.py)
re-scores a saved checkpoint **without retraining**, reporting both loss and
accuracy on the same three splits the training loop tracks:

```bash
python submissions/my_team/evaluate_model.py                       # default weights.joblib
python submissions/my_team/evaluate_model.py --weights path/to.joblib
python submissions/my_team/evaluate_model.py --tb                  # also log to a TensorBoard run dir
```

It reuses `train.py`'s exact transforms, split seed, and `evaluate()` so the
numbers line up with the run log, rebuilds the clean-val split with the same
`SEED`/`VAL_FRACTION`, prints an `acc | loss` table per split, and recomputes the
honest 50/50 estimate (`0.5·clean_val + 0.5·ood_test`). This is the
independent cross-check path: a checkpoint is verified here, separate from the
training loop, before any headline number is trusted (see
[debugging_notes.markdown](debugging_notes.markdown)).

## Sharing Results Across the Team

TensorBoard is a **local viewer**, not a shared service — `tensorboard --logdir runs` only shows the runs on *your own* machine. Since we work on separate laptops there's no shared `runs/` folder, and `runs/` is intentionally git-ignored (event files are large binaries that bloat history and merge-conflict). So the convention is:

1. **Commit the results table.** Record every meaningful run's config + scores in [exploration_and_tuning_report.markdown](exploration_and_tuning_report.markdown). This is the team's shared source of truth — small, diffable, and renders on GitHub.
2. **Zip-share the odd run** when a specific one is worth inspecting together. Each run dir is self-contained:
   ```bash
   zip -r run.zip submissions/my_team/runs/modelarchitecture_20260625_143000
   ```
   Drop it in the team chat / Drive; whoever wants to look unzips it and runs `tensorboard --logdir <unzipped path>` locally.

## OOD Evaluation Policy

Two augmentation types exist under `dataset/augmentations/`:

- **`color_jitter`** — used as a tuning signal; enters the selection score.
- **`random_rotation`** — frozen OOD test; evaluated once on the final selected checkpoint only, never seen during hyperparameter search.

Images whose clean originals appear in the augmentation set are held out of the training split entirely, preventing OOD leakage.

## Notes

- **Device** is auto-selected: CUDA → MPS → CPU.
- **MPS quirk:** host↔device copies use blocking transfers on purpose. A `non_blocking=True` copy of the labels can race the next kernel and feed garbage labels into the loss, silently collapsing training. Running stats are kept on-device and `.item()`-synced only once per epoch to avoid serializing MPS.
- **Optional dependency:** matplotlib + numpy enable the confusion-matrix and prediction-grid figures; everything else works without them.

<!-- fill in: hardware used, typical wall time -->
