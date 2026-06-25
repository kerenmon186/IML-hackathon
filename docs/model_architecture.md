# Model Architecture

## Overview

`ModelArchitecture` (defined in [submissions/my_team/model.py](../submissions/my_team/model.py)) is a from-scratch ResNet-18 adapted for a 20-class subset of ImageNet.

- **Input:** `[B, 3, 224, 224]` — ImageNet-normalized RGB images
- **Output:** `[B, 20]` — raw logits (no softmax)
- **Parameters:** 11,186,772 (~11.19 M) — standard ResNet-18 count, head reduced from 1000 → 20

## Layer-by-Layer Breakdown

### Stem

```
Conv2d(3, 64, 7x7, stride=2, padding=3, bias=False)   -> [B, 64, 112, 112]
BatchNorm2d(64) + ReLU
MaxPool2d(3x3, stride=2, padding=1)                    -> [B, 64, 56, 56]
```

The 7×7 stride-2 conv + stride-2 maxpool aggressively downsamples the input from 224 to 56, reducing memory and compute for the residual stages.

### Residual Stages

Each stage is built with `BasicBlock`s (two 3×3 convs + identity/projection shortcut). When the spatial resolution halves or channel count increases, a 1×1 projection shortcut aligns dimensions.

| Stage | Blocks | In channels | Out channels | Stride | Spatial size |
|-------|--------|-------------|--------------|--------|--------------|
| `layer1` | 2 | 64 | 64 | 1 | 56×56 |
| `layer2` | 2 | 64 | 128 | 2 | 28×28 |
| `layer3` | 2 | 128 | 256 | 2 | 14×14 |
| `layer4` | 2 | 256 | 512 | 2 | 7×7 |

### Head

```
AdaptiveAvgPool2d((1, 1))   -> [B, 512, 1, 1]
Flatten                      -> [B, 512]
Dropout(p=0.2)
Linear(512, 20)              -> [B, 20]
```

### BasicBlock

```
x ──> Conv3x3 ──> BN ──> ReLU ──> Conv3x3 ──> BN ──> (+) ──> ReLU
 └──────────────── (downsample if needed) ───────────────┘
```

## Initialization

- **Conv layers:** Kaiming Normal (`fan_out`, ReLU nonlinearity) — standard for ResNets trained from scratch.
- **BatchNorm:** weight=1, bias=0.
- **Last BN in each BasicBlock:** weight initialized to 0 (zero-init trick). This makes every block start as an identity, stabilizing early training.

## Design Rationale

The network is intentionally a plain, well-understood ResNet-18 trained from random initialization. ResNet-18's ~11 M parameters are a sensible capacity match for the ~20k-image training set — large enough to learn the 20 classes, small enough to avoid severe overfitting. Keeping the architecture deliberately standard means **robustness comes from the training strategy** (the light→heavy augmentation curriculum, label smoothing, weight decay, dropout) rather than from anything bespoke in the model, which keeps the design easy to reason about and the results easy to attribute.

The `layers` argument is parameterized `(2, 2, 2, 2)` so the same class can be respec'd to deeper ResNet variants (e.g. `(3, 4, 6, 3)` for ResNet-34) without code changes.

## Parameter Count

| Component | Parameters |
|-----------|------------|
| Stem (7×7 conv + BN) | 9,536 |
| `layer1` (64→64) | 147,968 |
| `layer2` (64→128) | 525,568 |
| `layer3` (128→256) | 2,099,712 |
| `layer4` (256→512) | 8,393,728 |
| `fc` (512→20) | 10,260 |
| **Total** | **11,186,772** |

The bulk of the parameters live in `layer4` (the 512-channel stage), as is typical for ResNets. The 20-class head is tiny (~10k params).
