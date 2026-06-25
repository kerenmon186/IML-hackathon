"""Post-run evaluation of a saved checkpoint.

Loads ``weights.joblib`` (or any state_dict path) and reports loss + accuracy on
the same three splits the training loop tracks, WITHOUT retraining:

    * clean-val           -- held-out slice of dataset/train (eval transform)
    * ood_val             -- color_jitter augmentation set (tuning signal)
    * ood_test            -- random_rotation augmentation set (frozen OOD-test)

The clean-val split is rebuilt with the same SEED/VAL_FRACTION as training, so
these numbers line up with the values logged during the run.

    python submissions/my_team/evaluate_model.py
    python submissions/my_team/evaluate_model.py --weights path/to/weights.joblib
    python submissions/my_team/evaluate_model.py --tb        # also log to TensorBoard
"""

import argparse
import time

import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import train as T  # reuse the exact transforms / splits / evaluate() from training


def evaluate_checkpoint(weights_path, batch_size=128, workers=0, to_tb=False):
    device = T.get_device()
    print(f"Device: {device}")

    eval_tf = T.build_eval_transform()
    _, val_idx = T.make_splits()
    val_full = T.ImageNetSubset(T.DATA_ROOT, "train", transform=eval_tf)
    val_set = Subset(val_full, val_idx)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=workers)

    ood_loaders = T.build_ood_loaders(eval_tf, batch_size, workers=workers)
    ood_val_loader = ood_loaders.get(T.OOD_VAL_NAME)
    ood_test_loader = ood_loaders.get(T.OOD_TEST_NAME)

    model = T.ModelArchitecture(num_classes=T.NUM_CLASSES).to(device)
    model.load_state_dict(joblib.load(weights_path))
    model.eval()
    # Match the training-time loss so values are directly comparable.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    results = {}
    with torch.no_grad():
        acc, loss = T.evaluate(model, val_loader, device, criterion)
        results["clean_val"] = (acc, loss)
        if ood_val_loader:
            results["ood_val"] = T.evaluate(model, ood_val_loader, device, criterion)
        if ood_test_loader:
            results["ood_test"] = T.evaluate(model, ood_test_loader, device, criterion)

    print(f"\nCheckpoint: {weights_path}")
    print(f"{'split':<24} {'acc':>8} {'loss':>8}")
    print("-" * 42)
    label = {
        "clean_val": "clean_val",
        "ood_val": f"ood_val ({T.OOD_VAL_NAME})",
        "ood_test": f"ood_test ({T.OOD_TEST_NAME})",
    }
    for key, (acc, loss) in results.items():
        print(f"{label[key]:<24} {acc:>8.3f} {loss:>8.3f}")

    if "clean_val" in results and "ood_test" in results:
        honest = 0.5 * results["clean_val"][0] + 0.5 * results["ood_test"][0]
        print(f"\nHonest 50/50 estimate (clean_val + ood_test): {honest:.3f}")

    if to_tb:
        from torch.utils.tensorboard import SummaryWriter
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = T.TB_LOGDIR / f"eval_{ts}"
        writer = SummaryWriter(log_dir=str(run_dir))
        for key, (acc, loss) in results.items():
            writer.add_scalar(f"eval/acc_{key}", acc, 0)
            writer.add_scalar(f"eval/loss_{key}", loss, 0)
        writer.flush()
        writer.close()
        print(f"\nTensorBoard: tensorboard --logdir {T.TB_LOGDIR}  (run: {run_dir.name})")

    return results


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default=str(T.OUTPUT),
                   help="Path to the state_dict to evaluate (default: weights.joblib).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--tb", action="store_true",
                   help="Also write the metrics to a TensorBoard run dir.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_checkpoint(args.weights, args.batch_size, args.workers, args.tb)
