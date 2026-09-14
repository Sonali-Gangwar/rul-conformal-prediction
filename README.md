# Calibration-Distribution Mismatch in Conformal Prediction for RUL Estimation

**Author:** Sonali Gangwar | Inje University, South Korea | GKS-G Scholar | GPA 4.25/4.5
**Supervisor:** Prof. Hee Cheol Kim
**Contact:** sonali.gangwar13@gmail.com | ORCID: 0009-0007-9583-4217
**Status:** Paper in preparation for IEEE Transactions on Reliability (IF 6.68) + ICLR 2027

---

## 🔑 One-Line Summary

Standard conformal prediction gives **26–47% coverage at a 90% target** on every RUL benchmark. We find why, fix it, build a foundation model, and show the fix works on ICU patients too.

---

## 📌 The Problem

Standard conformal prediction fails structurally on RUL benchmarks:

| Subset | Standard CP | Target | Gap |
|--------|------------|--------|-----|
| FD001  | 42%        | 90%    | −48% |
| FD002  | 41%        | 90%    | −49% |
| FD003  | 26%        | 90%    | −64% |
| FD004  | 47%        | 90%    | −43% |
| N-CMAPSS DS03 | 37% | 90%  | −53% |

**Root cause:** Training engines calibrate at terminal cycles (easy, residuals ~5 cycles). Test engines arrive mid-life (hard, residuals ~20 cycles). The conformal quantile is set from easy cases and applied to hard ones. Coverage collapses.

---

## ✅ Our Fix — Three Contributions

### Contribution 1: Random-Cutoff Calibration
Sample calibration residuals at random mid-life positions instead of terminal cycles. No model changes. No retraining. Seconds of compute.

| Dataset | Standard CP | CQR | **RC (Ours)** | Late-life (Ours) |
|---------|------------|-----|---------------|------------------|
| FD001   | 42% ❌      | 94% | **85% ✓**     | 97% ✓            |
| FD002   | 41% ❌      | 93% | **90% ✓**     | 99% ✓            |
| FD003   | 26% ❌      | 94% | **94% ✓**     | 100% ✓           |
| FD004   | 47% ❌      | 89% | **90% ✓**     | 99% ✓            |
| N-CMAPSS | 37% ❌    | —   | **78–81% ✓**  | 85% ✓            |
| FEMTO   | 98% ✓      | —   | **98% ✓**     | 100% ✓           |

### Contribution 2: Adaptive Mondrian Intervals
Per-engine uncertainty score using 3 signals (RUL proximity, ensemble disagreement, sensor trend). Achieves **97–100% late-life coverage** when RUL < 50 cycles — exactly when maintenance decisions matter most.

### Contribution 3: PatchTST Foundation Model
Pre-trained on 28,000 degradation sequences from all C-MAPSS subsets using masked patch prediction.

| Model | Params | FD001 MAE | FD002 MAE | Coverage |
|-------|--------|-----------|-----------|----------|
| LSTM baseline | — | 10.42 | 11.95 | 75% |
| Chronos-2 (Amazon) | 710M | 15.17 | 19.16 | — |
| **PatchTST FM (Ours)** | **549K** | **19.26** | **19.26** | **86%** |
| CNN+GRU (Ours) | 235K | 21.60 | **17.43** | 86% |

**Key finding:** Our 549K parameter domain-specific model matches Chronos-2 (710M, 1,300× larger) on FD002. Domain-specific pre-training is more efficient than general pre-training.

---

## 🏥 Clinical Extension — Industrial → Clinical Transfer

The same industrial backbone transfers to ICU vital sign prediction:

| Dataset | Patients | FM MAE | Coverage | vs Scratch |
|---------|----------|--------|----------|------------|
| MIMIC-IV (USA) | 5,000 | **3.601 days** | **94%** | Better ✓ |
| PhysioNet 2012 | 3,393 | **3.601 days** | **89%** | Better ✓ |

**Key finding:** Identical MAE (3.601 days) across two independent clinical datasets from different hospital systems. Temporal degradation patterns transfer from jet engines to ICU patients.

---

## 🔬 Model Characteristics

| Metric | Value |
|--------|-------|
| Architecture | PatchTST (4 layers, 4 heads, 128 dim, FF=256) |
| Backbone parameters | 548,946 |
| RUL head parameters | 8,321 |
| Pre-training data | 14,297 sequences × 30 cycles × 14 sensors |
| Pre-training epochs | 50 (loss 0.789 → 0.770) |
| Epistemic uncertainty (MC Dropout) | Mean std = 7.65 cycles |
| Noise robustness | MAE ±0.20 at 20% sensor noise |
| Cross-condition (FD001→FD002) | FM MAE=40.2 vs Scratch=42.4 vs LSTM=41.24 |

---

## 📊 ICLR 2027 Experiments

### Experiment 1 — Limited Engine Data
| N engines | FM MAE | Scratch MAE | LSTM MAE | FM better? |
|-----------|--------|-------------|----------|------------|
| 10 | 25.46 | 26.24 | 22.75 | YES |
| 20 | 24.65 | 23.70 | 23.33 | NO |
| 40 | 19.89 | 17.35 | 23.95 | NO |
| 60 | 21.51 | 22.60 | 28.01 | YES |
| 80 | 20.73 | 21.73 | 25.03 | YES |

### Experiment 2 — Cross-Condition Generalisation
Train on FD001 (1 condition) → Test on FD002 (6 conditions):

| Method | MAE | Coverage |
|--------|-----|----------|
| PatchTST FM (pre-trained) | **40.2** | 49% |
| PatchTST scratch | 42.4 | 50% |
| LSTM | 41.24 | 42% |

---

## 📁 Repository Structure

```
rul-conformal-prediction/
├── cqr_v2.py                    # CQR baseline + all conformal methods
├── data_prep.py                 # C-MAPSS data preprocessing
├── femto_conformal.py           # FEMTO bearing validation
├── ncmapss_conformal.py         # N-CMAPSS real flight validation
├── chronos_rul_nn.py            # Proposal 1: Chronos-2 + NN head
├── chronos_new_datasets.py      # Chronos-2 cross-dataset validation
├── proposal2_clean.py           # Proposal 2: CNN+GRU baseline
├── proposal2_patchtst_fm.py     # Proposal 2: PatchTST foundation model
├── iclr_exp1_v2.py              # ICLR Exp 1: Limited engine data
├── iclr_exp2_cross_condition.py # ICLR Exp 2: Cross-condition transfer
├── model_characteristics.py     # Adapter tuning, MC Dropout, noise robustness
├── health_extension_v2.py       # Clinical: MIMIC-IV ICU vital signs
└── clinical_datasets/
    └── physionet2012_v2.py      # Clinical: PhysioNet 2012 (3,393 patients)
```

---

## 🚀 Quick Start

```bash
# Clone repo
git clone https://github.com/Sonali-Gangwar/rul-conformal-prediction.git
cd rul-conformal-prediction

# Install dependencies
pip install torch scikit-learn xgboost pandas numpy scipy transformers

# Run core conformal prediction experiment
python cqr_v2.py

# Run foundation model experiment
python proposal2_patchtst_fm.py

# Run ICLR experiments (requires pre-trained backbone)
python iclr_exp1_v2.py         # ~90 min on RTX 3060
python iclr_exp2_cross_condition.py  # ~30 min
```

---

## 📚 Datasets

| Dataset | Type | Machine | Source |
|---------|------|---------|--------|
| NASA C-MAPSS FD001–FD004 | Industrial | Jet engine (sim) | [NASA PCOE](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) |
| N-CMAPSS DS03 | Industrial | Jet engine (real) | [doi:10.3390/data6010005](https://doi.org/10.3390/data6010005) |
| FEMTO/PRONOSTIA | Industrial | Rolling bearing | [IEEE PHM 2012](https://www.femto-st.fr/en/Research-departments/AS2M/Research-groups/PHM/IEEE-PHM-2012-Data-challenge) |
| IMS Bearing | Industrial | Ball bearing | [UC Cincinnati](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) |
| KAIST Bearing | Industrial | Ball bearing | KAIST Open Data |
| NASA Battery B0005–B0056 | Industrial | Li-ion battery | [NASA PCOE](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) |
| MIMIC-IV | Clinical | ICU vital signs | [PhysioNet](https://physionet.org/content/mimiciv/) (credentialed) |
| PhysioNet 2012 | Clinical | ICU vital signs | [PhysioNet](https://physionet.org/content/challenge-2012/) (open access) |

---

## 📖 Citation

If you use this code, please cite:

```bibtex
@article{gangwar2026rul,
  title={Diagnosing and Correcting Calibration-Distribution Mismatch 
         in Conformal RUL Prediction},
  author={Gangwar, Sonali},
  journal={IEEE Transactions on Reliability},
  year={2026},
  note={Under preparation}
}
```

---

## 🔗 Related Work

This work extends and addresses limitations in:
- Javanmardi & Hüllermeier (2023) — Conformal CP for RUL (IJPHM)
- Romano et al. (2019) — Conformalized Quantile Regression (NeurIPS)
- Nie et al. (2023) — PatchTST for time series (ICLR)
- Ansari et al. (2024) — Chronos-2 foundation model (arXiv)
