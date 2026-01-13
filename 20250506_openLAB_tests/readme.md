# Hybrid VAE–CNN/ML Framework for Structural Health Monitoring  
**openLAB Bridge – Open Case Study (TU Dresden)**

This repository provides a **complete, reproducible hybrid SHM pipeline** for distinguishing **sensor faults** from **structural faults** using experimental data from the **TU Dresden openLAB research bridge**.

The framework follows a **two-stage strategy**:

1. **Unsupervised anomaly detection** using an **LSTM-based Variational Autoencoder (VAE)** trained exclusively on *healthy data*.
2. **Supervised diagnostic classification** using a **CNN and classical machine-learning models** to identify the *type* of anomaly.

The repository is designed as an **open experimental case study** with **explicit assumptions**, suitable for **methodological research, benchmarking, and peer review**.

---

## 1. Scientific Motivation

In vibration-based Structural Health Monitoring (SHM), **anomalies are inevitable**, but their origin is ambiguous:

- **Structural fault**  
  (e.g. stiffness degradation, cracking, nonlinear response)
- **Sensor fault**  
  (e.g. drift, bias, spikes, noise, signal corruption)

This ambiguity is critical:  
misclassifying sensor faults as damage leads to **false alarms**, while missing real damage compromises **structural safety**.

To address this, the repository implements a **decoupled detection–diagnosis philosophy**:

- **Stage 1 – Detection (Unsupervised):**  
  Learn the distribution of *healthy structural behaviour only* using a VAE.

- **Stage 2 – Diagnosis (Supervised):**  
  Classify detected anomalies into **sensor fault** or **structural fault**.

This mirrors **real SHM practice**, where healthy data is abundant and damage labels are scarce.

---

## 2. Case Study: TU Dresden openLAB Research Bridge

**Data source**

> Herbers et al. (2025)  
> *Monitoring Data of the openLAB Research Bridge – Load Test on PE 2.1*  
> TU Dresden, Germany  
> DOI: https://doi.org/10.25532/OPARA-852

**Bridge summary**
- Prestressed concrete research bridge  
- Length: 45 m (3 × 15 m spans)  
- Dense instrumentation for displacement, strain, force, and temperature  
- Designed to produce both **structural damage** and **sensor faults** under controlled loading

**Data availability**
- Only **Day-2 load test (PE 2.1)** was available  
- Reference (Day-1) and tendon-cut (Day-3) data were **not accessible**  
- This necessitated **explicit and transparent labeling assumptions**

---

## 3. Sensor Channels Used in This Study (Explicit)

Although the dataset contains many channels, **not all sensors were used**.

### Selected channels
The analysis uses a **reduced, interpretable sensor subset**:

| Channel | Quantity | Usage |
|------|--------|------|
| Time | Time stamp | Synchronization |
| Strain gauge (1) | Local strain | Structural response indicator |
| Displacement sensor (1 per span) | Vertical displacement | Primary structural response |
| Load (resultant) | Applied load | Contextual information |

This results in:
- **One representative displacement channel per span**
- **One strain channel**
- **One load channel**
- **Time channel**

### Excluded channels
- Redundant displacement sensors at the same location  
- Environmental temperature channels  
- Sensors with excessive dropout or missing synchronization  

This selection ensures:
- Minimal redundancy  
- Physical interpretability  
- Consistent dimensionality across windows  

Exact channel mappings are documented in:
- `Data/raw/README_EN.md`
- `Data/raw/README_DE.md`

---

## 4. Dataset Generation and Labeling Rules

This repository **does not rely on hidden labels**.  
All class definitions are **explicit**.

### Raw data location

Data/raw/


| File | Max displacement |
|---|---|
| MD_2025_05_06_09_08_25.txt | 5 mm |
| MD_2025_05_06_10_43_20.txt | 10 mm |
| MD_2025_05_06_12_05_10.txt | 20 mm |
| MD_2025_05_06_13_43_17.txt | 30 mm |
| MD_2025_05_06_16_07_15.txt | 40 mm |
| MD_2025_05_06_17_39_40.txt | 50 mm |
| MD_2025_05_06_18_30_51.txt | 60 mm |

---

### Class definitions

#### **Normal (Healthy)**
- Load steps ≤ **20 mm**
- Elastic serviceability regime
- Stable sensors
- No visible damage

Used **exclusively** to train the **VAE**.

---

#### **Structural Fault**
- Load steps **30–60 mm**
- Physically consistent structural response
- Corrected measurements (after known sensor artefact removal)
- Interpreted as nonlinear response / stiffness degradation

---

#### **Sensor Fault**
Identified using **manufacturer documentation and experimental metadata**:
- Drift  
- Bias  
- Spikes  
- Noise bursts  
- Signal corruption during load plateaus  

Key point:
- **No synthetic fault injection**
- Faults originate from **real experimental artefacts**

---

## 5. Hybrid Pipeline Overview

Raw openLAB data
↓
Window extraction & labeling
↓
Run-based train / val / test split
↓
Feature construction
↓
VAE training (Normal only)
↓
Anomaly detection (VAE gate)
↓
Supervised classification (CNN / ML)
↓
Evaluation & plots


---

## 6. Repository Structure (Sketch)

20250506_openLAB_tests/
│
├── Codes/
│ ├── Models/
│ │ ├── temporal_vae_model.py # LSTM-VAE
│ │ └── cnn_model.py # CNN classifier
│ │
│ ├── 01_extract_windows_and_labels.py
│ ├── 02_make_splits.py
│ ├── 03_featurize_windows.py
│ ├── 04_train_vae.py
│ ├── 05_validate_vae.py
│ ├── 06_train_cnn.py
│ ├── 07_validate_cnn.py
│ ├── 08_train_ml_baselines.py
│ ├── 09_validate_ml_baselines.py
│ ├── 10_test_hybrid_pipeline.py
│ └── 11_generate_hybrid_pipeline_plot.py
│
├── Data/
│ └── raw/
│ ├── MD_*.txt
│ ├── README_EN.md
│ └── README_DE.md
│
├── Output/
│ ├── artifacts/
│ ├── figures/
│ └── metrics/
│
├── Photos/
└── README.md


---

## 7. Classifiers Used (Full Names)

**Deep learning**
- **CNN** – Convolutional Neural Network

**Classical machine learning**
- **CART** – Classification and Regression Tree  
- **RF** – Random Forest  
- **GB** – Gradient Boosting  
- **HGB** – Histogram-Based Gradient Boosting  
- **SVM-RBF** – Support Vector Machine with Radial Basis Function kernel  

All classifiers operate **only after anomaly detection by the VAE**.

---

## 8. How to Run the Pipeline

```bash
# 1. Data preparation
python Codes/01_extract_windows_and_labels.py
python Codes/02_make_splits.py
python Codes/03_featurize_windows.py

# 2. Train and validate VAE
python Codes/04_train_vae.py
python Codes/05_validate_vae.py

# 3. Train classifiers
python Codes/06_train_cnn.py
python Codes/08_train_ml_baselines.py

# 4. End-to-end evaluation
python Codes/10_test_hybrid_pipeline.py
python Codes/11_generate_hybrid_pipeline_plot.py

9. Outputs

Saved under Output/:

Row-normalized 3-class confusion matrices

Stage-2 metrics bar plots

Accuracy, Precision, Recall, F1-score, AUROC

10. Design Principles

Healthy-only training for anomaly detection

Explicit separation of detection and diagnosis

Run-based splits to avoid leakage

No synthetic fault injection

Transparent labeling assumptions

Reviewer-safe, reproducible workflow

@dataset{Herbers2025openLAB,
  author = {Herbers, Max and Richter, Bertram and Walker, Maria and Marx, Steffen},
  title  = {Monitoring Data of the openLAB Research Bridge – Load Test on PE 2.1},
  year   = {2025},
  doi    = {10.25532/OPARA-852}
}




