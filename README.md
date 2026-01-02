# hybrid-vae-cnn-for-shm

A reproducible hybrid structural health monitoring (SHM) workflow combining:
1) a **Temporal Variational Autoencoder (VAE)** for representation learning and anomaly gating, and  
2) a **Convolutional Neural Network (CNN)** for downstream classification where applicable (not used in the 1DOF interpretability experiment).

The repository supports three experiment tracks:

- **1DOF**: VAE interpretability (seen vs unseen variants; no CNN).
- **4DOF**: simulated multi-DOF structural response (normal vs structural fault).
- **openLAB**: field monitoring data from the openLAB Research Bridge (TU Dresden) with a hybrid VAE→CNN pipeline.

## Repository structure (high-level)

```text
configs/                     # experiment configurations (YAML)
src/                         # reusable modules (models, data, evaluation, plotting, utils)
experiments/
  1dof/                      # VAE interpretability only (no CNN)
  4dof/                      # simulated multi-DOF study
  openlab/                   # openLAB field monitoring pipeline
docs/                        # data statement, reproducibility guide, paper assets


## Quick start
1) Create environment

Python 3.10+ is recommended.

pip install -r requirements.txt


Optionally install in editable mode:

pip install -e .

2) Reproduce 1DOF experiment (VAE interpretability)

From repository root:

python -m experiments.1dof.scripts.00_generate_or_load
python -m experiments.1dof.scripts.01_train_vae
python -m experiments.1dof.scripts.02_validate_vae
python -m experiments.1dof.scripts.03_test_seen
python -m experiments.1dof.scripts.04_generate_unseen
python -m experiments.1dof.scripts.05_test_unseen


Outputs are written to:

experiments/1dof/outputs/

3) Reproduce openLAB experiment

This experiment requires downloading openLAB raw monitoring files (not redistributed here). See:

docs/data_statement.md

docs/reproducibility.md

Data availability

This repository does not redistribute third-party raw data. The openLAB dataset is available via TU Dresden’s archive and is licensed under CC BY-SA 4.0. See docs/data_statement.md for details and placement instructions.

Citation

If you use this repository, please cite it using CITATION.cff. For the openLAB dataset, also cite the original dataset DOI listed in docs/data_statement.md.