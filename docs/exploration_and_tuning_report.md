# Exploration & Tuning Report

## Scoring Metric

The grader scores on a 50/50 blend of in-distribution (clean) accuracy and
out-of-distribution (manipulated) accuracy, where the OOD half uses **hidden,
held-out augmentations not available to us**.

Locally we approximate this with:

```
selection_score = 0.5 * clean_val + 0.5 * ood_val      # ood_val = color_jitter (the tuning proxy)
honest_estimate = 0.5 * clean_val + 0.5 * ood_test      # ood_test = random_rotation (FROZEN, never tuned)
```

`color_jitter` is the OOD signal we are allowed to tune against, so it enters
the selection score. `random_rotation` is held frozen — evaluated once on the
final checkpoint only — so the *honest_estimate* is our best unbiased guess at
the grader's OOD half (a manipulation family we never trained or tuned on).

---

## Architecture Exploration

We used a plain, from-scratch **ResNet-18**. The design choice was deliberate:
~11.19M parameters is a sensible capacity match for ~18k training images, and
keeping the architecture standard means robustness is attributable to the
*training strategy*, not to bespoke architecture tricks. Deeper variants are a
one-argument change (`layers=(3,4,6,3)` → ResNet-34) but were **not run** — the
ResNet-18 baseline already reached ~0.92 and we prioritized the robustness study.

| Architecture | Params | Clean val | OOD val (color_jitter) | Score | Notes |
|--------------|--------|-----------|------------------------|-------|-------|
| **ResNet-18** (chosen) | 11.19M | 0.923 | 0.940 | 0.931 | from scratch; submission model |
| ResNet-34 | ~21M | — | — | — | considered, not run (overfitting risk on 18k imgs) |

---

## Augmentation Exploration

This is the core of the project. Two strategies were run end-to-end (50 epochs,
batch 128, identical otherwise). Both use the **light → heavy curriculum**:
train with gentle geometry (crop + flip) until clean-val crosses 0.60, then
switch on the heavy set. Rotation is never trained on (it is the frozen test).

| Strategy | Clean val | color_jitter | **rotation (frozen)** | honest | Notes |
|----------|-----------|--------------|------------------------|--------|-------|
| **Curriculum, base heavy aug** (chosen) | 0.923 | 0.940 | **0.920** | **0.921** | ColorJitter + RandomGrayscale + RandomErasing |
| Curriculum, *enriched* heavy aug | 0.918 | 0.950 | **0.910** | 0.914 | + GaussianBlur, AdjustSharpness, Autocontrast, Equalize, Posterize, harsher crop/colour |

**Result: enriching the augmentation did NOT help and was not merged.** It
improved the manipulation we *trained toward* (color_jitter 0.940 → 0.950) but
**regressed the frozen, unseen `random_rotation`** (0.920 → 0.910) and clean
accuracy (0.923 → 0.918). The deltas are small (~1% on 400-image OOD sets, near
noise), so the honest reading is "no real generalization gain, slight added
risk + complexity." Critically, if we had selected on the *tunable* signal we
would have picked the enriched model (selection score 0.934 > 0.931) — the
frozen test is exactly what stopped us from doing that.

### Extended augmentation push (overnight) — diminishing returns

We didn't stop at one enriched run. Over an overnight session we kept pushing
augmentation diversity further, progressively stacking and tuning additional
transforms on top of the base heavy set — Gaussian blur, sharpness adjustment,
autocontrast, histogram equalization, posterization, stronger colour jitter, and
more aggressive crop scales. The pattern was consistent across these runs:
**diminishing returns**. Each addition, at best, nudged the trained-toward
`color_jitter` metric up by fractions of a percent, while the frozen
`random_rotation` test stayed flat-to-down and clean accuracy eroded slightly.
Past the base heavy-aug set, extra augmentation kept trading in-distribution
accuracy for no honest OOD gain — we were optimizing the signal we could see
rather than the generalization we actually wanted. Having converged on that
conclusion, we stopped enriching and shipped the simpler base curriculum, which
held the best honest 50/50 estimate (0.921).

---

## Hyperparameter Tuning

Hyperparameters were chosen from standard from-scratch ResNet practice and not
swept (the baseline recipe worked on the first correct run; effort went into the
robustness study instead). Documented here for completeness and the interview.

### Learning Rate

| LR | Schedule | Score | Notes |
|----|----------|-------|-------|
| **0.1** (chosen) | 3-epoch warmup + cosine to 0 | 0.931 | classic SGD-momentum ResNet recipe; not swept |

Per-step (not per-epoch) cosine so the schedule is batch-size invariant.

### Batch Size

Batch size **128** was chosen from a throughput benchmark, not an accuracy
sweep: on the M-series GPU compute was flat at ~300–315 img/s for 64/128/256
(GPU-bound, no OOM), so 128 was picked for the better BatchNorm/gradient
estimates at no speed cost. Full epoch ≈ 58–75 s; 50 epochs ≈ 60 min.

### Augmentation Threshold (`--aug-threshold`)

| Threshold | Switch epoch (observed) | Score | Notes |
|-----------|-------------------------|-------|-------|
| **0.6** (chosen) | epoch 5 (clean-val 0.607) | 0.931 | not swept; forced fallback at 40% of epochs if never reached |

---

## Regularization

| Config | Score | Notes |
|--------|-------|-------|
| **dropout=0.2, wd=5e-4, label-smoothing=0.1** (chosen) | 0.931 | standard set; not ablated individually |

Train acc ≈ 0.96 vs clean-val ≈ 0.92 → a small ~4–5 pt gap with val still rising
at the end, i.e. the model generalizes rather than memorizes.

---

## Key Findings

1. **The curriculum closes the clean-vs-OOD gap.** Before the light→heavy switch
   the OOD sets lagged clean accuracy by ~5 points; after heavy augmentation
   turned on (epoch 5) the gap closed to ~1–2 points, and OOD often slightly
   *exceeded* clean accuracy. Building features clean-first, then teaching
   invariance, beat augmenting hard from step 0.

2. **More augmentation ≠ more robustness.** Adding blur/sharpness/autocontrast/
   equalize/posterize raised the *trained-toward* color_jitter but *regressed*
   the unseen rotation and clean accuracy. Augmentation diversity can overfit to
   the manipulations you can see; it does not automatically transfer to unseen
   ones.

3. **A frozen OOD type is essential for honest selection.** Selecting on the
   tunable signal (color_jitter) would have chosen the worse model. Holding
   `random_rotation` out — never trained, never tuned, evaluated once — is what
   gives a trustworthy estimate of the grader's hidden OOD half.

4. **The result is verified, not a metric artifact.** Numbers were reproduced
   through the grader's own `predict.py` path (independent of the training loop),
   and predictions are spread across all 20 classes (worst-class accuracy ~0.75–
   0.79), ruling out class collapse. (We earlier caught a real bug — an MPS
   `non_blocking=True` label-copy race that silently corrupted training — which
   is why every headline number is cross-checked.)

---

## Final Configuration

```bash
python submissions/my_team/train.py \
    --epochs 50 \
    --batch-size 128 \
    --workers 4 \
    --lr 0.1 \
    --weight-decay 5e-4 \
    --label-smoothing 0.1 \
    --aug-threshold 0.6 \
    --force-switch-frac 0.4
```

**Final scores (selected checkpoint, epoch 49):**

| Split | Accuracy |
|-------|----------|
| Clean val | 0.923 |
| OOD val (color_jitter, tuning proxy) | 0.940 |
| OOD test (random_rotation, frozen) | 0.920 |
| Selection score (0.5·clean + 0.5·color_jitter) | 0.931 |
| **Honest 50/50 estimate (0.5·clean + 0.5·rotation)** | **0.921** |
