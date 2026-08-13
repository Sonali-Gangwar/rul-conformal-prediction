# Calibration-Distribution Mismatch in Conformal Prediction for RUL Estimation

**A Random-Cutoff Calibration Strategy with Adaptive Mondrian Intervals**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Dataset: C-MAPSS](https://img.shields.io/badge/Dataset-NASA%20C--MAPSS-green.svg)](https://www.nasa.gov/intelligent-systems-division/)

---

## Overview

This repository contains the full implementation of my research paper:

> **"Calibration-Distribution Mismatch in Conformal Prediction for Remaining Useful Life Estimation: A Random-Cutoff Calibration Strategy with Adaptive Intervals"**
>
> *Targeting: IEEE Transactions on Reliability (IF 6.68)*

### The Research Problem

Standard conformal prediction (CP) applied to RUL prediction achieves only **28–48% empirical coverage** at a 90% nominal level — far below the theoretical guarantee. The root cause is a **calibration-distribution mismatch**: training engines run to failure (last-cycle calibration = easy, small residuals), while test engines are truncated at unknown mid-life points (hard, large residuals).

### My Contributions

1. **Identification** — First formal characterisation of the calibration-distribution mismatch in conformal RUL prediction, measured across all four C-MAPSS subsets.

2. **Random-Cutoff Calibration** — Proposed calibrating at random mid-life cutoff points instead of last cycles, recovering **85–94% coverage** across all subsets.

3. **Adaptive Mondrian Conformal Intervals** — Per-engine personalised quantile using three uncertainty signals (RUL proximity, model disagreement, sensor trend), achieving **97–100% late-life coverage** (RUL < 50 cycles).

---

## Results Summary

### Coverage Comparison (Target = 90%)

| Subset | Standard CP | Random-Cutoff (proposed) | Adaptive Mondrian (proposed) |
|--------|-------------|--------------------------|------------------------------|
| FD001  | 40%         | 85%                      | 86%                          |
| FD002  | 48%         | 93%                      | 92%                          |
| FD003  | 31%         | 94%                      | 95%                          |
| FD004  | 45%         | 94%                      | 93%                          |

### Late-Life Coverage (RUL < 50 cycles) — Most Important

| Subset | Standard CP | Random-Cutoff | Adaptive Mondrian |
|--------|-------------|---------------|-------------------|
| FD001  | 61%         | 94%           | **97%**           |
| FD002  | 81%         | 100%          | **99%**           |
| FD003  | 52%         | 100%          | **100%**          |
| FD004  | 67%         | 99%           | **99%**           |

---

## Repository Structure

```
rul_project/
│
├── data/                          # Place C-MAPSS data files here
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   └── ... (FD002, FD003, FD004)
│
├── data_prep.py                   # Step 1: Load + preprocess FD001
├── train_baseline.py              # Step 2: Train RF + XGB + LSTM baseline
├── conformal_v5.py                # Step 3: Global random-cutoff conformal
├── adaptive_conformal_v2.py       # Step 4: Adaptive Mondrian conformal
├── run_all_subsets.py             # Run all 4 FD subsets (main experiment)
├── ablation_study.py              # Ablation study
├── paper_results_table.py         # Generate all paper tables
│
└── README.md                      # This file
```

---

## How to Run

### 1. Get the Data

Download the NASA C-MAPSS dataset from [Kaggle](https://www.kaggle.com/datasets/palbha/cmapss-jet-engine-simulated-data) and place all `.txt` files in the `data/` folder.

### 2. Install Requirements

```bash
pip install numpy pandas scikit-learn xgboost torch
```

### 3. Run the Full Pipeline

```bash
# Step 1: Preprocess
python data_prep.py

# Step 2: Train baseline models
python train_baseline.py

# Step 3: Run all 4 subsets with all 3 conformal methods
python run_all_subsets.py

# Step 4: Adaptive Mondrian conformal
python adaptive_conformal_v2.py

# Step 5: Generate paper tables
python paper_results_table.py
```

---

## Key Concept: What is Conformal Prediction?

Conformal prediction is a method that **wraps around any trained model** and converts its single-number output into a guaranteed prediction interval. It does not assume any distribution shape — it learns the interval width from actual past prediction errors.

**Example:** Instead of "this engine has 80 cycles left," it says "80 ± 21 cycles, and I guarantee the true value falls inside this range at least 90% of the time."

The guarantee holds **only if** calibration and test data come from the same distribution. This work identifies that this assumption is violated in standard RUL datasets and proposes a fix.

---

## Papers Referenced

1. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer.
2. Javanmardi, A., & Hüllermeier, E. (2023). Conformal Prediction Intervals for Remaining Useful Lifetime Estimation. *IJPHM*, 14(2). [arXiv:2212.14612](https://arxiv.org/abs/2212.14612)
3. Nie, Y., et al. (2023). A Time Series is Worth 64 Words. *ICLR 2023*. [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
4. Vovk, V., et al. (2003). Mondrian Confidence Machine. Royal Holloway Technical Report.
5. Saxena, A., & Goebel, K. (2008). Turbofan Engine Degradation Simulation Data Set. NASA Ames.

---

## Author

**Sonali Gangwar**
Master's Student, Computer Engineering
Inje University, Gimhae, South Korea
Collaboration: University of Florida (Prof. Nam-Ho Kim)

---

## Citation

If you use this code, please cite:

```bibtex
@article{gangwar2026calibration,
  title={Calibration-Distribution Mismatch in Conformal Prediction for
         Remaining Useful Life Estimation: A Random-Cutoff Calibration
         Strategy with Adaptive Intervals},
  author={Gangwar, Sonali and Kim, Hee Cheol and Kim, Nam-Ho},
  journal={IEEE Transactions on Reliability},
  year={2026},
  note={Under review}
}
```
