# Exploration & Tuning Report

## Scoring Metric

The grader scores on a 50/50 blend of clean accuracy and OOD accuracy:

```
score = 0.5 * clean_val + 0.5 * ood_val
```

All experiments are compared on this composite score. The frozen `random_rotation` OOD test is reported separately for honest estimation only.

---

## Architecture Exploration

<!-- What architectures were tried? (e.g., ResNet-18, ResNet-34, EfficientNet, pretrained vs from-scratch) -->
<!-- What was the motivation for each? What did you observe? -->

| Architecture | Params | Clean val | OOD val | Score | Notes |
|--------------|--------|-----------|---------|-------|-------|
| ResNet-18 (current) | ~11.19M | — | — | — | baseline |
| | | | | | |

---

## Augmentation Exploration

<!-- What augmentation strategies were tried beyond the current curriculum? -->
<!-- e.g., MixUp, CutMix, AutoAugment, TrivialAugment, stronger ColorJitter, RandAugment -->

| Strategy | Clean val | OOD val | Score | Notes |
|----------|-----------|---------|-------|-------|
| Light → Heavy curriculum (current) | — | — | — | baseline |
| | | | | |

---

## Hyperparameter Tuning

### Learning Rate

<!-- What LR values were tried? What schedule variants? -->

| LR | Schedule | Score | Notes |
|----|----------|-------|-------|
| 0.1 (current) | warmup-3 + cosine | — | baseline |
| | | | |

### Batch Size

<!-- How did batch size interact with LR? -->

### Augmentation Threshold (`--aug-threshold`)

<!-- Did changing the phase-switch threshold matter? -->

| Threshold | Switch epoch (typical) | Score | Notes |
|-----------|------------------------|-------|-------|
| 0.6 (current) | — | — | baseline |
| | | | |

---

## Regularization

<!-- Dropout strength, weight decay, label smoothing experiments -->

| Config | Score | Notes |
|--------|-------|-------|
| dropout=0.2, wd=5e-4, ls=0.1 (current) | — | baseline |
| | | |

---

## Key Findings

<!-- fill in: most impactful changes, surprising results, things that did not help -->

1.
2.
3.

---

## Final Configuration

<!-- Summarize the chosen hyperparameters and why -->

```bash
python submissions/my_team/train.py \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.1 \
    --weight-decay 5e-4 \
    --label-smoothing 0.1 \
    --aug-threshold 0.6 \
    --force-switch-frac 0.4
```

**Final scores:**

| Split | Accuracy |
|-------|----------|
| Clean val | — |
| OOD val (color_jitter) | — |
| OOD test (random_rotation, frozen) | — |
| Composite (50/50 estimate) | — |
