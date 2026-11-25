# IML Hackathon: Robust 20-Class Image Classification

A ResNet-18 trained from scratch (PyTorch) to classify images into 20 classes and stay accurate under image manipulations it never saw during training. Built as a team project for the Introduction to Machine Learning (IML) hackathon at the Hebrew University of Jerusalem.

**Team:** David Kaluta, Dana Kimelman, Keren Monsonego, Shaima Zbeda

## Results

| Evaluation | Accuracy |
|---|---|
| Clean validation | **0.924** |
| Color jitter (used for tuning, out-of-distribution) | **0.940** |
| Random rotation (frozen, never trained on) | **0.935** |

Rotation was kept out of training on purpose and evaluated once, on the selected checkpoint, so its accuracy is an honest estimate of how the model handles unseen manipulations. It stays essentially equal to clean accuracy.

## Approach

**Model.** ResNet-18 (BasicBlocks, stages `[2,2,2,2]`, widths 64/128/256/512), trained from random initialization, outputting 20 logits for a `3x224x224` ImageNet-normalized input. About 11M parameters suit roughly 18k training images without heavy overfitting. Additions: zero-initialization of the last BatchNorm in each residual block, Kaiming initialization, and dropout (0.2) before the classifier.

**Training.**
- Split: 90% train / 10% clean validation, stratified with a fixed seed. Every training image that has a provided augmented counterpart is forced into validation, so the out-of-distribution check never overlaps with training data.
- Optimizer: SGD (momentum 0.9, Nesterov, weight decay 5e-4)
- Schedule: 3-epoch linear warmup, then cosine decay over 50 epochs
- Loss: cross-entropy with label smoothing 0.1
- Batch size 128, input 224x224, ImageNet mean/std normalization
- Validation transforms match the grader's pipeline exactly (Resize 256, CenterCrop 224, ToTensor, Normalize)

**Augmentation curriculum.** Robustness comes from the training strategy rather than the architecture. Aggressive augmentation from step 0 slows down a from-scratch network, so training starts with light geometric augmentation only (random resized crop and flip). Once clean validation accuracy passes 0.60, the full set is switched on: color jitter, random grayscale and random erasing. In our run the switch happened at epoch 5. A forced fallback at 40% of the epochs guarantees the switch even if the threshold is never reached.

Before the switch, the out-of-distribution sets lagged clean accuracy by about 5 points. After it, the gap closed to 1-2 points. The train/validation gap stayed small (about 5 points) with validation still improving at the end, so the model generalizes rather than memorizes.

**Checkpoint selection.** Best score of `0.5 * clean-val + 0.5 * OOD-val`, mirroring the 50/50 grading. Predictions were verified through the grader's own `predict.py` path and are spread across all 20 classes.

## Repository structure

- `submissions/my_team/` - the team's submission and write-up
- `colab.ipynb` - Google Colab notebook
- `requirements.txt`, `pyproject.toml` - dependencies (Python 3.13)

## Setup

```bash
pip install -r requirements.txt
```

Training can also be run in Google Colab with `colab.ipynb`.

## Tech

Python, PyTorch, ResNet-18, TensorBoard
