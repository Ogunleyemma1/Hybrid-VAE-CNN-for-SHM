"""
experiments/1dof/scripts/00_generate_or_load.py

Generate the SEEN dataset (normal/drifted/noisy), window it, and split at RUN-level
to avoid leakage.

Outputs:
- datasets/raw/seen/*.npz
- datasets/processed/X_{train,val,test}.npy
- datasets/processed/y_{train,val,test}.npy
- datasets/processed/meta_{train,val,test}.csv
- datasets/processed/run_split.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.seed import set_seed  # OK to reuse your src utils
from scripts._helpers.sim_1dof import SimConfig, make_seen_signal
from scripts._helpers.windowing_1d import windowize_1d
from scripts._helpers.io_local import ensure_dir, save_npz, save_csv, save_json


def _run_level_split(run_ids: list[int], seed: int, train: float, val: float, test: float):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(run_ids).tolist()

    n = len(perm)
    n_train = max(1, int(round(train * n)))
    n_val = max(1, int(round(val * n)))
    n_train = min(n_train, n - 2)
    n_val = min(n_val, n - n_train - 1)

    train_ids = sorted(perm[:n_train])
    val_ids = sorted(perm[n_train : n_train + n_val])
    test_ids = sorted(perm[n_train + n_val :])
    return train_ids, val_ids, test_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--train_ratio", type=float, default=0.4)
    ap.add_argument("--val_ratio", type=float, default=0.3)
    ap.add_argument("--test_ratio", type=float, default=0.3)
    ap.add_argument("--n_normal", type=int, default=40)
    ap.add_argument("--n_drifted", type=int, default=20)
    ap.add_argument("--n_noisy", type=int, default=20)
    args = ap.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    root = Path("experiments/1dof/datasets")
    raw_seen = root / "raw" / "seen"
    processed = root / "processed"
    ensure_dir(raw_seen)
    ensure_dir(processed)

    sim = SimConfig()
    n_steps = int(sim.t_total / sim.dt) + 1

    # 1) Generate runs (transparent raw artifacts for reviewers)
    run_table = []
    run_id = 0
    for label, n_runs in [("normal", args.n_normal), ("drifted", args.n_drifted), ("noisy", args.n_noisy)]:
        for _ in range(n_runs):
            out = make_seen_signal(sim, n_steps, label=label, rng=rng)
            fname = f"run_{run_id:04d}_{label}.npz"
            save_npz(raw_seen / fname, **out)
            run_table.append({"run_id": run_id, "label": label, "file": fname})
            run_id += 1

    runs_df = pd.DataFrame(run_table)
    save_csv(processed / "runs_seen.csv", runs_df)

    # 2) Windowing
    X_all, y_all, meta_rows = [], [], []
    for r in run_table:
        d = np.load(raw_seen / r["file"])
        x = d["x"].astype(np.float32)

        Xw, starts = windowize_1d(x, window=args.window, stride=args.stride)
        if Xw.shape[0] == 0:
            continue

        X_all.append(Xw)
        y_all.append(np.array([r["label"]] * Xw.shape[0], dtype=object))

        for s in starts:
            meta_rows.append(
                {"run_id": r["run_id"], "label": r["label"], "source_file": r["file"], "start_idx": int(s)}
            )

    X = np.concatenate(X_all, axis=0).astype(np.float32)
    y = np.concatenate(y_all, axis=0).astype(object)
    meta = pd.DataFrame(meta_rows)

    # 3) Run-level split (prevents leakage)
    run_ids = sorted(runs_df["run_id"].unique().tolist())
    tr_ids, va_ids, te_ids = _run_level_split(run_ids, args.seed, args.train_ratio, args.val_ratio, args.test_ratio)
    save_json(processed / "run_split.json", {"seed": args.seed, "train_runs": tr_ids, "val_runs": va_ids, "test_runs": te_ids})

    def write_split(name: str, run_set: set[int]) -> None:
        m = meta["run_id"].isin(run_set).to_numpy()
        np.save(processed / f"X_{name}.npy", X[m])
        np.save(processed / f"y_{name}.npy", y[m])
        meta.loc[m].reset_index(drop=True).to_csv(processed / f"meta_{name}.csv", index=False)

    write_split("train", set(tr_ids))
    write_split("val", set(va_ids))
    write_split("test", set(te_ids))


if __name__ == "__main__":
    main()
