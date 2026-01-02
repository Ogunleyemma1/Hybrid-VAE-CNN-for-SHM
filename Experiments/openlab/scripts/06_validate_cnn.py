# experiments/openlab/scripts/06_validate_cnn.py
"""
Step 06: Validate CNN on VAL(GOLD) or TEST(GOLD) SF/E using frozen threshold.

Inputs:
  - processed/X_raw.npy
  - processed/window_labels_augmented.csv
  - processed/run_split.json
  - outputs/models/cnn_openlab.pt
  - outputs/models/cnn_norm_stats.npz
  - outputs/metrics/cnn_threshold.npy

Outputs:
  - outputs/metrics/cnn_eval_summary_<split>_gold.json
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


def apply_standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip: float = 10.0) -> np.ndarray:
    x = (X - mu[None, None, :]) / sd[None, None, :]
    x = np.clip(x, -clip, clip)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)


def encode_sf_e(labels: pd.Series) -> np.ndarray:
    lab = labels.astype(str).to_numpy()
    if not np.all(np.isin(lab, ["SF", "E"])):
        bad = sorted(set(lab) - set(["SF", "E"]))
        raise ValueError(f"Unexpected labels: {bad}")
    return np.where(lab == "SF", 0, 1).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="experiments/openlab/datasets/processed")
    ap.add_argument("--split", type=str, choices=["val", "test"], default="val")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.4)
    args = ap.parse_args()

    root = find_repo_root()
    processed_dir = resolve_under_root(args.processed_dir, root=root)

    dirs = default_experiment_dirs("experiments/openlab")
    logger = configure_logging(name="openlab", log_file=dirs["logs"] / "06_validate_cnn.log")

    X = np.load(processed_dir / "X_raw.npy").astype(np.float32)
    meta = pd.read_csv(processed_dir / "window_labels_augmented.csv")
    splits = json.loads((processed_dir / "run_split.json").read_text(encoding="utf-8"))

    eval_runs = set(splits["val_runs"] if args.split == "val" else splits["test_runs"])
    is_se = meta["label"].isin(["SF", "E"])
    mask = meta["run_id"].isin(eval_runs) & is_se & (meta["label_source"] == "gold")
    if int(mask.sum()) == 0:
        raise RuntimeError(f"No {args.split.upper()}(GOLD) SF/E windows available.")

    X_eval_raw = X[mask.to_numpy()]
    y_eval = encode_sf_e(meta.loc[mask, "label"])

    # load norm + thr
    norm = np.load(dirs["models"] / "cnn_norm_stats.npz")
    mu = norm["mean"].astype(np.float32)
    sd = norm["std"].astype(np.float32)
    thr = float(np.load(dirs["metrics"] / "cnn_threshold.npy").ravel()[0])

    X_eval = apply_standardize(X_eval_raw, mu, sd)
    Xt = torch.tensor(X_eval[:, None, :, :], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(input_channels=1, num_classes=2, dropout_rate=args.dropout).to(device)
    model.load_state_dict(torch.load(dirs["models"] / "cnn_openlab.pt", map_location=device))
    model.eval()

    probs_e = []
    with torch.no_grad():
        for i in range(0, Xt.shape[0], args.batch_size):
            xb = Xt[i:i+args.batch_size].to(device)
            logits = model(xb)
            p = torch.softmax(logits, dim=1)[:, 1]
            probs_e.append(p.detach().cpu().numpy())
    probs_e = np.concatenate(probs_e, axis=0)

    y_pred = (probs_e >= thr).astype(int)

    rep = classification_report(y_eval, y_pred, target_names=["SF", "E"], zero_division=0)
    cm = confusion_matrix(y_eval, y_pred)

    summary = {
        "split": args.split,
        "runs": sorted(list(eval_runs)),
        "n_eval": int(len(y_eval)),
        "n_sf": int((y_eval == 0).sum()),
        "n_e": int((y_eval == 1).sum()),
        "threshold": float(thr),
        "confusion_matrix": cm.tolist(),
    }
    out_path = dirs["metrics"] / f"cnn_eval_summary_{args.split}_gold.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(rep)
    logger.info(f"CM:\n{cm}")
    logger.info(f"Saved: {out_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
