# 4DOF Hybrid VAE–CNN Framework for Structural Health Monitoring

This repository contains a complete and reproducible implementation of a **hybrid anomaly detection and fault diagnosis framework** for **Structural Health Monitoring (SHM)** based on a **four-degree-of-freedom (4DOF) structural dynamic system**.

The framework combines:
- an **unsupervised Temporal Variational Autoencoder (TVAE)** to model normal structural behaviour and detect anomalies, and
- a **supervised Convolutional Neural Network (CNN)** to classify detected anomalies into **sensor faults** or **structural faults**.

The implementation is fully script-driven and follows a clearly defined data-generation, training, validation, and testing workflow.

---

## 1. Structural System

The simulated structure is a **4DOF lumped-mass spring–damper system**.  
Each simulation produces **12-channel time-series data**:

- Displacements: \( x_1, x_2, x_3, x_4 \)
- Velocities: \( v_1, v_2, v_3, v_4 \)
- Accelerations: \( a_1, a_2, a_3, a_4 \)

All datasets share the same:
- excitation signal,
- numerical integration scheme,
- time discretisation.

This ensures that differences between datasets arise from **fault mechanisms**, not numerical artefacts.

---

## 2. Dataset Description

### 2.1 Normal (Healthy) Data
Generated using nominal system parameters.

- Used exclusively for:
  - training the Temporal VAE,
  - validating the VAE,
  - computing normalization statistics.

### 2.2 Structural Fault Data
Structural faults are simulated as **global stiffness reductions** of varying severity (e.g. 8%, 18%, 30%, 40%).

- Affect all state channels simultaneously.
- Represent physically meaningful damage scenarios.

### 2.3 Sensor Fault Data
Sensor faults are **localized to a single degree of freedom**.

For a given faulty sensor \( i \), the fault affects:
- displacement \( x_i \),
- velocity \( v_i \),
- acceleration \( a_i \),

while all other channels remain unchanged.

Implemented sensor fault types include:
- noise amplification,
- signal drift,
- constant bias,
- impulsive spikes.

This design reflects realistic measurement faults encountered in SHM systems.

---

## 3. Repository Structure

```text
4DOF/
│
├── Scripts/
│   ├── Models/
│   │   ├── temporal_vae.py        # Temporal VAE architecture
│   │   └── cnn_model.py           # CNN fault classifier
│   │
│   ├── utils/
│   │   ├── simulation_4dof.py     # 4DOF system simulation
│   │   └── windowing.py           # Sliding-window utilities
│   │
│   ├── 00_make_run_splits.py      # Run-level dataset splits
│   ├── 01_generate_normal_runs.py
│   ├── 02_generate_fault_datasets.py
│   ├── 03_train_vae.py
│   ├── 04_vae_thresholding.py
│   ├── 05_train_cnn.py
│   └── 06_test_full_pipeline.py
│
├── Data/
│   ├── raw/
│   │   ├── normal/
│   │   └── faults/
│   │       ├── sensor_fault/
│   │       └── structural_fault/
│   │
│   └── processed/
│       ├── run_splits.json
│       ├── normal_stats.npz
│       ├── vae_threshold.json
│       ├── stage1_vae_train_meta.json
│       └── stage2_cnn_train_meta.json
│
├── models/
│   ├── temporal_vae_state_dict.pt
│   └── cnn_state_dict.pt
│
└── Output/
    └── figures/


## 4. Data Splitting Strategy

All datasets are split at the **run level** into:

- **40% training**
- **30% validation**
- **30% testing**

The splits are fixed and stored in:

Data/processed/run_splits.json


---

## 5. Stage 1 – Temporal VAE (Anomaly Detection)

### Training

```bash
 - python -m Scripts.03_train_vae

Trained only on normal training runs

Validated on normal validation runs

Learns a low-dimensional latent representation of healthy system dynamics

Reconstruction error (MSE) is used as the anomaly score

Threshold Selection
python -m Scripts.04_vae_thresholding


Threshold derived from the validation reconstruction error distribution

Saved to:

Data/processed/vae_threshold.json


Windows with reconstruction error above the threshold are treated as anomalous

## 6. Stage 2 – CNN Fault Diagnosis
Input Representation

For each anomalous window, the CNN input consists of two channels:

Normalized signal window

Squared reconstruction residual from the VAE

Input shape:

(N, 2, window_length, 12)

Training
python -m Scripts.05_train_cnn


Trained only on anomalous windows from fault datasets

Binary classification:

Sensor Fault

Structural Fault

Validation used for early stopping and model selection

## 7. Full Pipeline Evaluation
python -m Scripts.06_test_full_pipeline

Evaluation Procedure

Test windows are passed through the VAE

Normal windows are classified as Normal

Anomalous windows are routed to the CNN

CNN outputs are mapped to:

Sensor Fault

Structural Fault

Outputs

Window-level three-class accuracy

Confusion matrices (counts and row-normalized)

Anomaly rates per class

Stored metrics and figures

Class Labels
0 = Normal
1 = Sensor Fault
2 = Structural Fault

### 8. Key Characteristics

Clear separation between anomaly detection and fault diagnosis

VAE trained exclusively on healthy data

Sensor faults are localized

Structural faults are global

Sliding-window processing ensures temporal consistency

Fully reproducible, script-based workflow

### 9. Intended Applications

This framework is intended for:

Methodological research in data-driven Structural Health Monitoring

Evaluation of hybrid unsupervised–supervised diagnostic pipelines

Extension to higher-fidelity simulations or experimental structures
