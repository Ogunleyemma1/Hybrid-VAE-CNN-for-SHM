## Overview — 1DOF Temporal VAE Study

This directory contains a **single-degree-of-freedom (1DOF) benchmark study** for validating a **Temporal Variational Autoencoder (TVAE)** applied to structural time-series data.  
The 1DOF system provides a controlled and physically interpretable baseline for verifying signal generation, temporal encoding, latent-space organization, and reconstruction-based error behavior prior to extension to multi-DOF systems.

---

## Directory Structure and Script Overview

```text
1_DOF/
│
├── Data/
│   ├── raw/            # Generated time-series data
│   └── processed/      # Normalization statistics and metadata
│
├── Scripts/
│   ├── Models/         # TVAE model definition
│   ├── datasets.py     # Windowing, normalization, RMSE utilities
│   ├── signals_1dof.py # 1DOF signal generation utilities
│   │
│   ├── 01_generate_seen_variants.py
│   ├── 02_generate_unseen_variants.py
│   ├── 03_train_vae.py
│   ├── 04_test_seen_variants.py
│   ├── 05_test_unseen_variants.py
│   └── 06_compare_seen_vs_unseen_rmse.py
│
├── Output/
│   ├── figures/        # Publication-ready figures
│   └── tables/         # CSV summaries and statistics
│
└── models/
    └── temporal_vae_state_dict.pt


### Key Components

- **Signal generation**
  - *Seen data*: physically consistent free-vibration responses of a 1DOF oscillator (Newmark–β integration)
  - *Unseen data*: analytically defined signals used exclusively for generalization assessment

- **Model training**
  - TVAE trained only on seen data
  - Deterministic 50/50 temporal train–test split
  - KL divergence regularization with annealing

- **Latent-space analysis**
  - Direct visualization of raw latent coordinates (true learned dimensions)
  - PCA of latent means to reveal dominant learned structures
  - Latent points colored by signal variant for interpretability

- **Reconstruction evaluation**
  - Window-wise reconstruction with stitching
  - Segment-wise RMSE as the primary quantitative metric
  - Separate evaluation on seen and unseen datasets

- **Comparative assessment**
  - Direct comparison of RMSE trends and distributions between seen and unseen data
  - Clear indication of reconstruction error sensitivity under distribution shift

### Reproducibility

- Normalization statistics computed only from training data  
- Train/test split stored explicitly and reused across scripts  
- All figures saved in **PDF, PNG, and SVG** formats  
- No manual post-processing required

### Purpose of the 1DOF Study

The 1DOF benchmark establishes confidence in the **temporal modeling capability**, **latent representation quality**, and **reconstruction-based anomaly sensitivity** of the TVAE architecture, providing a validated foundation for subsequent multi-DOF structural health monitoring studies.
