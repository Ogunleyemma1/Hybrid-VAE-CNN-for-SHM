# experiments/4dof/scripts/00_generate_or_load.py
"""
Step 00 (4DOF): Generate or validate raw simulation datasets.

Creates / expects:
  experiments/4dof/datasets/raw/normal/*.csv
  experiments/4dof/datasets/raw/faults/sensor_faults/**/*.csv
  experiments/4dof/datasets/raw/faults/structural_faults/**/*.csv

Optionally runs legacy generators (your current scripts) and copies outputs into this structure.

Rationale:
- Keep raw data generation separate from windowing/training for auditability.
- Ensure a clean, reviewer-friendly dataset directory layout.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from src.utils import configure_logging, default_experiment_dirs, find_repo_root, resolve_under_root


def _count_csv(p: Path) -> int:
    return len(list(p.rglob("*.csv")))


def _run_py(script_path: Path, cwd: Path) -> None:
    if not script_path.is_file():
        raise FileNotFoundError(f"Missing legacy script: {script_path}")
    subprocess.check_call([sys.executable, str(script_path)], cwd=str(cwd))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="experiments/4dof/datasets/raw")
    ap.add_argument("--force_regen", action="store_true")
    ap.add_argument(
        "--use_legacy_generators",
        action="store_true",
        help="Run legacy generator scripts then copy outputs into repo structure.",
    )
    ap.add_argument(
        "--legacy_root",
        type=str,
        default=".",
        help="Folder containing legacy generator scripts and their output directories.",
    )
    args = ap.parse_args()

    root = find_repo_root()
    raw_dir = resolve_under_root(args.raw_dir, root=root)

    dirs = default_experiment_dirs("experiments/4dof")
    logger = configure_logging(name="4dof", log_file=dirs["logs"] / "00_generate_or_load.log")

    normal_dir = raw_dir / "normal"
    sensor_dir = raw_dir / "faults" / "sensor_faults"
    struct_dir = raw_dir / "faults" / "structural_faults"

    for p in [normal_dir, sensor_dir, struct_dir]:
        p.mkdir(parents=True, exist_ok=True)

    if args.force_regen:
        logger.info("force_regen enabled: clearing raw dataset folders.")
        for p in [normal_dir, sensor_dir, struct_dir]:
            if p.exists():
                shutil.rmtree(p)
                p.mkdir(parents=True, exist_ok=True)

    existing = _count_csv(raw_dir)
    if existing > 0 and not args.force_regen:
        logger.info(f"Raw dataset already present. CSV count={existing} under {raw_dir}")
        logger.info(f"Normal CSVs: {_count_csv(normal_dir)}")
        logger.info(f"Sensor-fault CSVs: {_count_csv(sensor_dir)}")
        logger.info(f"Structural-fault CSVs: {_count_csv(struct_dir)}")
        logger.info("Done.")
        return

    if not args.use_legacy_generators:
        logger.error(
            "Raw dataset folders are empty.\n"
            "Either:\n"
            "  (A) rerun with --use_legacy_generators, or\n"
            "  (B) place generated CSVs manually into experiments/4dof/datasets/raw/.\n"
            "Expected:\n"
            "  raw/normal/*.csv\n"
            "  raw/faults/sensor_faults/**/*.csv\n"
            "  raw/faults/structural_faults/**/*.csv\n"
        )
        raise SystemExit(2)

    legacy_root = resolve_under_root(args.legacy_root, root=root)
    logger.info(f"Running legacy generators in: {legacy_root}")

    # You uploaded these generator scripts; here we assume they're in legacy_root:
    # - generate_healthy_datasets.py
    # - generate_variety_fault_data.py
    _run_py(legacy_root / "generate_healthy_datasets.py", cwd=legacy_root)
    _run_py(legacy_root / "generate_variety_fault_data.py", cwd=legacy_root)

    # Your legacy generators are assumed to write to:
    # legacy_root/data_generation/healthy_runs and legacy_root/data_generation/faults
    legacy_normal = legacy_root / "data_generation" / "healthy_runs"
    legacy_faults = legacy_root / "data_generation" / "faults"

    if not legacy_normal.is_dir() or _count_csv(legacy_normal) == 0:
        raise RuntimeError(f"Legacy normal output not found/empty: {legacy_normal}")
    if not legacy_faults.is_dir() or _count_csv(legacy_faults) == 0:
        raise RuntimeError(f"Legacy faults output not found/empty: {legacy_faults}")

    # Copy normal -> raw/normal
    for f in legacy_normal.glob("*.csv"):
        shutil.copy2(f, normal_dir / f.name)

    # Copy faults preserving one folder level for readability
    legacy_sensor = legacy_faults / "sensor_faults"
    legacy_struct = legacy_faults / "structural_faults"

    if legacy_sensor.is_dir():
        for f in legacy_sensor.rglob("*.csv"):
            out = sensor_dir / f.parent.name
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out / f.name)

    if legacy_struct.is_dir():
        for f in legacy_struct.rglob("*.csv"):
            out = struct_dir / f.parent.name
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out / f.name)

    logger.info(f"Raw dataset populated at: {raw_dir}")
    logger.info(f"Normal CSVs: {_count_csv(normal_dir)}")
    logger.info(f"Sensor-fault CSVs: {_count_csv(sensor_dir)}")
    logger.info(f"Structural-fault CSVs: {_count_csv(struct_dir)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
