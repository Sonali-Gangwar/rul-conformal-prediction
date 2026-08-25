# Calibration-Distribution Mismatch in Conformal Prediction for RUL Estimation

**Paper:** Under preparation for IEEE Transactions on Reliability  
**Author:** Sonali Gangwar, Inje University, South Korea (GKS-G Scholar)

## Problem
Standard conformal prediction achieves only **26–47% empirical coverage at a 90% nominal target** on NASA C-MAPSS. We identify the cause: a calibration-distribution mismatch — training engines are calibrated at their last cycle (easy) while test engines are truncated at unknown mid-life points (hard).

## Solution
1. **Random-cutoff calibration** — recovers 85–94% coverage on C-MAPSS, 78–81% on N-CMAPSS real flight data
2. **Adaptive Mondrian intervals** — 97–100% late-life coverage (RUL < 50 cycles)

## Cross-dataset validation
| Dataset | Standard CP | Our method | Machine type |
|---|---|---|---|
| C-MAPSS FD001–FD004 | 26–47% | 85–94% | Jet engine (sim) |
| N-CMAPSS DS03 | 37% | 78–81% | Jet engine (real) |
| FEMTO Bearing | 98% | 98% | Rolling bearing |

## Scripts
| Script | Description |
|---|---|
| `cqr_v2.py` | CQR baseline (Tables 3 & 4 in paper) |
| `chronos_rul_nn.py` | Proposal 1: Chronos-2 + NN head |
| `chronos_ncmapss.py` | Proposal 1: N-CMAPSS cross-dataset |
| `chronos_femto.py` | Proposal 1: FEMTO bearing |
| `chronos_new_datasets.py` | Proposal 1: IMS, Battery, KAIST |
| `proposal2_clean.py` | Proposal 2: CNN+GRU vs Chronos-2 |

## Requirements