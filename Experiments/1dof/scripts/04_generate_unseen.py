"""
experiments/1dof/scripts/04_generate_unseen.py

Generate UNSEEN variants and window them (no splitting).
Outputs:
- datasets/raw/unseen/*.npz
- datasets/processed/X_unseen.npy
- datasets/processed/y_unseen.npy
- datasets/processed/meta_unseen.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.seed import set_seed
from scripts._helpers.sim_1dof import SimConfig, make_unseen_signal
from scripts._helpers.windowing_1d import windowize_1d
from scripts._helpers.io_local import ensure_dir, save_npz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--n_runs_per_variant", type=int, default=15)
    args = ap.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    root = Path("experiments/1dof/datasets")
    raw_unseen = root / "raw" / "unseen"
    processed = root / "processed"
    ensure_dir(raw_unseen)
    ensure_dir(processed)

    sim = SimConfig()
    n_steps = int(sim.t_total / sim.dt) + 1

    variants = ["amp_shifted", "freq_shifted", "noise_heavy", "drift_heavy"]

    run_table = []
    rid = 0
    for var in variants:
        for _ in range(args.n_runs_per_variant):
            out = make_unseen_signal(sim, n_steps=n_steps, variant=var, rng=rng)
            fname = f"unseen_{rid:04d}_{var}.npz"
            save_npz(raw_unseen / fname, **out)
            run_table.append({"run_id": rid, "label": var, "file": fname})
            rid += 1

    # window
    X_all, y_all, meta_rows = [], [], []
    for r in run_table:
        d = np.load(raw_unseen / r["file"])
        x = d["x"].astype(np.float32)

        Xw, starts = windowize_1d(x, window=args.window, stride=args.stride)
        if Xw.shape[0] == 0:
            continue

        X_all.append(Xw)
        y_all.append(np.array([r["label"]] * Xw.shape[0], dtype=object))

        for s in starts:
            meta_rows.append({"run_id": r["run_id"], "label": r["label"], "source_file": r["file"], "start_idx": int(s)})

    X = np.concatenate(X_all, axis=0).astype(np.float32)
    y = np.concatenate(y_all, axis=0).astype(object)
    meta = pd.DataFrame(meta_rows)

    np.save(processed / "X_unseen.npy", X)
    np.save(processed / "y_unseen.npy", y)
    meta.to_csv(processed / "meta_unseen.csv", index=False)


if __name__ == "__main__":
    main()
