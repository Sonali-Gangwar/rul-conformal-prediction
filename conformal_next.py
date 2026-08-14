"""
conformal_next.py  —  STEP 2 PREVIEW (your contribution)

This is a STUB, not the full method yet. It shows exactly where the conformal
interval code plugs in, using the artefacts saved by train_baseline.py.

The idea (split conformal prediction, the simplest version):
  1. You already held out a CALIBRATION set in train_baseline.py and saved the
     absolute residuals  r_i = |y_i - pred_i|  to outputs/calib_residuals.npy
  2. For a target coverage of 90%, find the 90th-percentile residual  q.
     (small finite-sample correction: use ceil((n+1)*0.9)/n quantile)
  3. The prediction interval for ANY new point is:  [pred - q, pred + q]
  4. CHECK CALIBRATION: on the test set, what fraction of true RUL values actually
     fall inside the interval? If ~90%, the interval is honest (well-calibrated).

Run (after train_baseline.py):  python conformal_next.py
"""

import os
import numpy as np

OUT_DIR = "outputs"
TARGET_COVERAGE = 0.90


def main():
    resid = np.load(os.path.join(OUT_DIR, "calib_residuals.npy"))
    preds = np.load(os.path.join(OUT_DIR, "test_predictions.npz"))
    lstm_pred = preds["lstm"]
    y_test = preds["y_test"]

    # --- split conformal quantile with finite-sample correction ---
    n = len(resid)
    level = np.ceil((n + 1) * TARGET_COVERAGE) / n
    level = min(level, 1.0)
    q = np.quantile(resid, level)

    lower = lstm_pred - q
    upper = lstm_pred + q

    inside = (y_test >= lower) & (y_test <= upper)
    coverage = inside.mean()
    avg_width = (upper - lower).mean()

    print("=" * 55)
    print(f"  Split-conformal (target {TARGET_COVERAGE:.0%})")
    print("-" * 55)
    print(f"  calibration residual quantile q : {q:8.2f}")
    print(f"  empirical coverage on test      : {coverage:8.1%}")
    print(f"  average interval width          : {avg_width:8.2f}")
    print("=" * 55)
    print("\n  -> Fill these into the 'Conformal' row of your notes table.")
    print("  -> Next: variance & confidence intervals, then the")
    print("     extrapolation-aware variant (coverage near end-of-life).")


if __name__ == "__main__":
    main()
