# Running on Google Colab

The project runs on Colab GPU unchanged — `train.py` auto-selects the device
(`cuda` → `mps` → `cpu`) and uses only repo-relative paths. Use **`colab.ipynb`**
(at the repo root) as a ready-made runner, or follow the steps below.

Open the notebook directly:
`https://colab.research.google.com/github/dkaluta/huji-iml-hackathon/blob/main/colab.ipynb`
(Colab will prompt to authorize GitHub access since the repo is private.)

> Set the Colab runtime to **GPU**: Runtime → Change runtime type → T4/A100 GPU.

## 1. Clone the repo (private → needs a token)
```python
TOKEN = ""  # GitHub PAT with 'repo' scope (Settings → Developer settings → tokens)
!git clone https://{TOKEN}@github.com/dkaluta/huji-iml-hackathon.git
%cd huji-iml-hackathon
```

## 2. Dependencies
Colab already has torch/torchvision/numpy/pillow/matplotlib/tensorboard built
for its CUDA driver — **don't reinstall torch.** Only `joblib` may be missing:
```python
!pip install -q joblib
```

## 3. Dataset
The dataset is git-ignored (too large), so place it under `dataset/`. The repo
ships `dataset/labels.json`; you just add `train/` and `augmentations/`.

**Recommended — mount Drive and symlink** (no copying of 20k files):
```python
from google.colab import drive
drive.mount('/content/drive')
# point these at your copy of the course folders in Drive:
!ln -sfn "/content/drive/MyDrive/iml_hackathon/train"          dataset/train
!ln -sfn "/content/drive/MyDrive/iml_hackathon/augmentations"  dataset/augmentations
!ls dataset && ls dataset/train | head
```
Expected layout: `dataset/train/<class>/*.jpg` and
`dataset/augmentations/{color_jitter,random_rotation}/<class>/*.jpg`.

## 4. Train
```python
!python submissions/my_team/train.py --epochs 50 --batch-size 128 --workers 2
```
(`workers 2` is safe on Colab; bump if the instance has more cores. On a T4 a
full 50-epoch run is roughly comparable to the local M-series timing.)

## 5. Verify the submission format
```python
!python check_submission.py my_team
```

## 6. TensorBoard (optional)
```python
%load_ext tensorboard
%tensorboard --logdir submissions/my_team/runs
```

## 7. Download the trained weights
```python
from google.colab import files
files.download('submissions/my_team/weights.joblib')
```

## Notes
- **Experiment scripts** work too: `submissions/my_team/kfold_cv.py` (k-fold CV)
  and `submissions/my_team/ood_eval.py` (synthetic OOD robustness panel).
- The MPS `non_blocking` caveat in `docs/debugging_notes.markdown` does not apply
  on CUDA, but the fix (plain `.to(device)`) is harmless there.