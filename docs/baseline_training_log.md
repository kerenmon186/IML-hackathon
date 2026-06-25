# Baseline Training Log (submission model)

This is the run that produced the submitted `weights.joblib` (ResNet-18,
light→heavy augmentation curriculum), trained 2026-06-24 on Apple M4 Pro (MPS).

```bash
python submissions/my_team/train.py --epochs 50 --batch-size 128 --workers 4
```

> **Note on completeness.** The raw `train.log` from this run was overwritten by
> a later experiment run (training logs are not version-controlled), and the
> TensorBoard run dir was cleared before the next run. The lines below are the
> epoch numbers captured live while monitoring the run, so a few stretches
> (epochs 9–28 and 44) were not recorded and are marked as gaps — **not**
> interpolated or fabricated. The final selection and frozen-OOD numbers are the
> exact reported values, independently re-verified through `predict.py`.

```
Device: mps
Loaded 20000 images from dataset/train
Split: 18000 train / 2000 val (791 val images held out to protect the OOD probe)
Loaded 400 images from dataset/augmentations/color_jitter
Loaded 400 images from dataset/augmentations/random_rotation
Curriculum: light aug -> heavy at clean-val>=0.60 or epoch 20 (whichever first)

epoch   1/50 | lr 0.0336 | phase 1 | train_loss 2.542 acc 0.255 | val 0.356 | ood_val 0.338 | score 0.347
epoch   2/50 | lr 0.0669 | phase 1 | train_loss 2.164 acc 0.404 | val 0.479 | ood_val 0.440 | score 0.459
epoch   3/50 | lr 0.1000 | phase 1 | train_loss 1.963 acc 0.489 | val 0.541 | ood_val 0.493 | score 0.517
epoch   4/50 | lr 0.0999 | phase 1 | train_loss 1.739 acc 0.575 | val 0.531 | ood_val 0.475 | score 0.503
epoch   5/50 | lr 0.0996 | phase 1 | train_loss 1.550 acc 0.653 | val 0.607 | ood_val 0.570 | score 0.588
  -> [curriculum] HEAVY augmentation ON at epoch 5 (val>=thresh, val=0.607)
epoch   6/50 | lr 0.0990 | phase 2 | train_loss 1.792 acc 0.552 | val 0.605 | ood_val 0.570 | score 0.588
epoch   7/50 | lr 0.0982 | phase 2 | train_loss 1.631 acc 0.621 | val 0.677 | ood_val 0.642 | score 0.660
epoch   8/50 | lr 0.0972 | phase 2 | train_loss 1.542 acc 0.654 | val 0.720 | ood_val 0.700 | score 0.710

      ... epochs 9–28 not captured (steady mid-training climb) ...

epoch  29/50 | lr 0.0417 | phase 2 | train_loss 0.935 acc 0.881 | val 0.854 | ood_val 0.848 | score 0.851
epoch  30/50 | lr 0.0384 | phase 2 | train_loss 0.907 acc 0.890 | val 0.822 | ood_val 0.832 | score 0.827
epoch  31/50 | lr 0.0352 | phase 2 | train_loss 0.898 acc 0.895 | val 0.867 | ood_val 0.868 | score 0.867
epoch  32/50 | lr 0.0320 | phase 2 | train_loss 0.885 acc 0.899 | val 0.844 | ood_val 0.855 | score 0.850
epoch  33/50 | lr 0.0290 | phase 2 | train_loss 0.861 acc 0.908 | val 0.872 | ood_val 0.882 | score 0.877
epoch  34/50 | lr 0.0260 | phase 2 | train_loss 0.835 acc 0.919 | val 0.887 | ood_val 0.892 | score 0.889
epoch  35/50 | lr 0.0231 | phase 2 | train_loss 0.824 acc 0.924 | val 0.879 | ood_val 0.890 | score 0.885
epoch  36/50 | lr 0.0203 | phase 2 | train_loss 0.815 acc 0.925 | val 0.892 | ood_val 0.908 | score 0.900
epoch  37/50 | lr 0.0177 | phase 2 | train_loss 0.797 acc 0.932 | val 0.903 | ood_val 0.902 | score 0.903
epoch  38/50 | lr 0.0152 | phase 2 | train_loss 0.776 acc 0.939 | val 0.898 | ood_val 0.897 | score 0.898
epoch  39/50 | lr 0.0129 | phase 2 | train_loss 0.767 acc 0.944 | val 0.908 | ood_val 0.918 | score 0.913
epoch  40/50 | lr 0.0108 | phase 2 | train_loss 0.751 acc 0.950 | val 0.901 | ood_val 0.915 | score 0.908
epoch  41/50 | lr 0.0088 | phase 2 | train_loss 0.735 acc 0.956 | val 0.919 | ood_val 0.923 | score 0.921
epoch  42/50 | lr 0.0070 | phase 2 | train_loss 0.725 acc 0.960 | val 0.910 | ood_val 0.920 | score 0.915
epoch  43/50 | lr 0.0054 | phase 2 | train_loss 0.717 acc 0.964 | val 0.914 | ood_val 0.933 | score 0.923

      ... epoch 44 not captured ...

epoch  45/50 | lr 0.0028 | phase 2 | train_loss 0.703 acc 0.970 | val 0.919 | ood_val 0.935 | score 0.927
epoch  46/50 | lr 0.0018 | phase 2 | train_loss 0.697 acc 0.972 | val 0.920 | ood_val 0.938 | score 0.929
epoch  47/50 | lr 0.0010 | phase 2 | train_loss 0.696 acc 0.972 | val 0.922 | ood_val 0.938 | score 0.930
epoch  48/50 | lr 0.0004 | phase 2 | train_loss 0.693 acc 0.973 | val 0.919 | ood_val 0.942 | score 0.931
epoch  49/50 | lr 0.0001 | phase 2 | train_loss 0.685 acc 0.976 | val 0.923 | ood_val 0.940 | score 0.931
epoch  50/50 | lr 0.0000 | phase 2 | train_loss 0.688 acc 0.974 | val 0.923 | ood_val 0.938 | score 0.930

Selected epoch 49: clean_val 0.923 | ood_val (color_jitter) 0.940
FROZEN OOD-test (random_rotation): 0.920  -> honest 50/50 estimate 0.921
Saved best weights to submissions/my_team/weights.joblib
Reload check OK: weights.joblib matches ModelArchitecture.
```

## Reading the log

- **Epochs 1–3:** LR warms up (0.034 → 0.10); clean-val climbs 0.36 → 0.54.
- **Epoch 5:** clean-val crosses the 0.60 threshold → curriculum switches to heavy
  augmentation (`phase 1 → 2`).
- **Epoch 6:** the expected post-switch blip — train acc dips (0.65 → 0.55) as the
  harder augmentation kicks in; clean-val holds.
- **Epochs 7+:** fast recovery; the OOD–clean gap closes (pre-switch OOD lagged
  ~5 pts; afterward it tracks within ~1–2 pts and sometimes exceeds clean).
- **End:** train ≈ 0.97 vs clean-val ≈ 0.92 — a small ~5-pt gap with val still
  rising → generalizing, not memorizing. Best checkpoint = epoch 49.
