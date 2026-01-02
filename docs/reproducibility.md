# Reproducibility

## Scope

This document describes how to reproduce the results produced by this repository:

- Synthetic experiments: **1DOF** and **4DOF**
- Field monitoring experiment: **openLAB Research Bridge**

The overall workflow is:

1. Create/obtain datasets
2. Extract windows and create splits
3. Train models (Temporal VAE; optionally CNN for classification where applicable)
4. Validate and evaluate metrics
5. Generate manuscript-ready figures under `docs/paper_assets/`

---

## Environment setup

### Python version

Python 3.10+ is recommended.

### Install dependencies

From repository root:

```bash
pip install -r requirements.txt
