# experiments/4dof/scripts/06_validate_cnn.py
"""
Step 06 (4DOF): Evaluate CNN on VAL or TEST runs for SF vs ST.

Outputs:
  outputs/metrics/cnn_eval_report_<split>.txt
  outputs/metrics/cnn_eval_summary_<split>.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.models.cnn_model import CNN
from src.utils import configure_logging, default_experiment_dirs, find_repo_root, resolve_under_root


def standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip: float = 10.0) -> np.ndarray:
    Z = (X - mu[None, None, :]) / sd[None, None, :]
    Z = np.clip(Z, -clip, clip)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z.astype(np.float32)


def encode_sf_st(labels: pd.Series) -> np.ndarray:
    lab = labels.astype(str).to_numpy()
    return np.where(lab == "SF", 0, 1).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="experiments/4dof/datasets/processed")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.4)
    args = ap.parse_args()

    root = find_repo_root()
    processed_dir = resolve_under_root(args.processed_dir, root=root)
    dirs = default_experiment_dirs("experiments/4dof")
    logger = configure_logging(name="4dof", log_file=dirs["logs"] / "06_validate_cnn.log")

    X = np.load(processed_dir / "X.npy").astype(np.float32)
    meta = pd.read_csv(processed_dir / "meta_windows.csv")
    splits = json.loads((processed_dir / "run_split.json").read_text(encoding="utf-8"))

    eval_runs = set(splits["val_runs"] if args.split == "val" else splits["test_runs"])
    mask = meta["run_id"].isin(eval_runs) & meta["label"].isin(["SF", "ST"])

    Xe = X[mask.to_numpy()]
    ye = encode_sf_st(meta.loc[mask, "label"])

    stats = np.load(dirs["models"] / "cnn_norm_stats.npz")
    mu = stats["mean"].astype(np.float32)
    sd = stats["std"].astype(np.float32)
    Ze = standardize(Xe, mu, sd)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(input_channels=1, num_classes=2, dropout_rate=args.dropout).to(device)
    model.load_state_dict(torch.load(dirs["models"] / "cnn_4dof.pt", map_location=device))
    model.eval()

    Xt = torch.tensor(Ze[:, None, :, :], dtype=torch.float32)
    y_pred = []
    with torch.no_grad():
        for i in range(0, Xt.shape[0], args.batch_size):
            xb = Xt[i:i+args.batch_size].to(device)
            y_pred.extend(model(xb).argmax(1).detach().cpu().numpy().tolist())

    rep = classification_report(ye, y_pred, target_names=["SF", "ST"], zero_division=0)
    cm = confusion_matrix(ye, y_pred)

    (dirs["metrics"] / f"cnn_eval_report_{args.split}.txt").write_text(rep, encoding="utf-8")
    summary = {
        "split": args.split,
        "runs": sorted(list(eval_runs)),
        "n_eval": int(len(ye)),
        "confusion_matrix": cm.tolist(),
    }
    (dirs["metrics"] / f"cnn_eval_summary_{args.split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(rep)
    logger.info(f"CM:\n{cm}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
