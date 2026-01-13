# Hybrid LSTM–VAE and CNN Framework for Structural Health Monitoring  
**Joint Diagnosis of Sensor Faults and Structural Faults**

## Overview

This repository contains the complete, reproducible implementation of a **hybrid deep-learning framework for Structural Health Monitoring (SHM)** that integrates:

- an **LSTM-based Variational Autoencoder (LSTM–VAE)** for **unsupervised anomaly detection**, and  
- **supervised classifiers** (CNN and classical machine-learning models) for **fault attribution**, distinguishing  
  **sensor faults** from **structural faults**.

In vibration-based SHM, deviations from normal behaviour may arise either from **true structural degradation** or from **sensor malfunction**. Conventional approaches often conflate these effects, leading to false alarms or misdiagnosis.  
This work proposes a **staged diagnostic pipeline** that first detects anomalies in a healthy-only learning setting and subsequently classifies their physical origin.

The methodology follows a **progressive three-stage validation strategy**, moving from controlled numerical simulation to real-world experimental data, consistent with the accompanying manuscript.

---

## Repository Structure

```text
Hybrid-VAE-CNN-for-SHM/
│
├── 1_DOF/                          # Stage 1: Fundamental validation (1-DOF system)
│   ├── Data/
│   │   ├── raw/                    # Generated signal variants (seen and unseen excitations)
│   │   └── processed/              # Metadata and train/validation/test splits
│   ├── Scripts/
│   │   ├── Models/                 # LSTM–VAE architecture (temporal autoencoder)
│   │   ├── 01_generate_seen_variants.py
│   │   ├── 02_generate_unseen_variants.py
│   │   ├── 03_train_vae.py
│   │   ├── 04_test_seen_variants.py
│   │   ├── 05_test_unseen_variants.py
│   │   └── 06_compare_seen_vs_unseen_rmse.py
│   └── readme.md
│
├── 4DOF/                           # Stage 2: Hybrid diagnosis in a multi-DOF system
│   ├── Data/
│   │   ├── raw/                    # Normal, sensor-fault, and structural-fault datasets
│   │   └── processed/              # Run-level splits, thresholds, and training metadata
│   ├── Scripts/
│   │   ├── Models/                 # LSTM–VAE and CNN models
│   │   ├── utils/                  # Simulation, windowing, preprocessing utilities
│   │   ├── 00_make_run_splits.py
│   │   ├── 01_generate_normal_runs.py
│   │   ├── 02_generate_fault_datasets.py
│   │   ├── 03_train_vae.py
│   │   ├── 04_vae_thresholding.py
│   │   ├── 05_train_cnn.py
│   │   └── 06_test_full_pipeline.py
│   ├── pyproject.toml
│   └── readme.md
│
├── 20250506_openLAB_tests/          # Stage 3: Experimental validation (Open-Lab Bridge)
│   ├── Codes/
│   │   ├── Models/                 # CNN and LSTM–VAE adapted to experimental data
│   │   ├── 01_extract_windows_and_labels.py
│   │   ├── 02_make_splits.py
│   │   ├── 03_featurize_windows.py
│   │   ├── 04_train_vae.py
│   │   ├── 05_validate_vae.py
│   │   ├── 06_train_cnn.py
│   │   ├── 07_validate_cnn.py
│   │   ├── 08_train_ml_baselines.py
│   │   ├── 09_validate_ml_baselines.py
│   │   ├── 10_test_hybrid_pipeline.py
│   │   └── 11_generate_hybrid_pipeline_plot.py
│   ├── Data/
│   │   ├── extracted/              # Window labels, run diagnostics, and splits
│   │   └── raw/                    # Raw experimental logs (not used directly for training)
│   └── readme.md
│
├── .gitattributes                  # Enforces LF line endings for reproducibility
├── .gitignore                      # Excludes heavy artifacts and generated outputs
└── README.md                       # This file
