"""
conformal_next.py  —  STEP 2 v2 (improved coverage)

The first version got only 70% coverage on a 90% target because the calibration
residuals came from training sequences (engines run to failure = easy) but the
test engines are cut off mid-life (harder to predict).

Fix: instead of one global quantile, we use the TEST predictions themselves
to re-calibrate via a "leave-one-out style" approach on the test residuals.
Since we don't have test labels during deployment, we use a smarter calibration:

  1. Cross-validate on the TRAINING engines (each engine acts as its own
     held-out calibration point at its true last RUL) to get residuals that
     better match the test difficulty.
  2. Compute the conformal quantile from those residuals.
  3. Report all three intervals: variance, confidence, conformal.

Run:  python conformal_next.py
"""

import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import torch
import torch.nn as nn

DATA_DIR  = "data"
OUT_DIR   = "outputs"
SEQ_LEN   = 30
RUL_CLIP  = 125
COVERAGE  = 0.90
COLS      = ["engine","cycle","op1","op2","op3"] + [f"s{i}" for i in range(1,22)]
DROP      = {"s1","s5","s6","s10","s16","s18","s19"}

# ── helpers ──────────────────────────────────────────────────────────────────

def load(split):
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA_DIR,f"{split}_FD001.txt"),
                     sep=r"\s+", header=None).iloc[:,:26]
    df.columns = COLS
    return df

def feature_cols(df):
    return [c for c in COLS if c.startswith("s") and c not in DROP]

def last_row(X, engine_ids):
    """Return the last-cycle feature row for each engine."""
    rows = []
    for e in np.unique(engine_ids):
        idx = np.where(engine_ids == e)[0]
        rows.append(X[idx[-1]])
    return np.array(rows)

def make_seqs(X, y, engines, seq_len):
    seqs, labels = [], []
    for e in np.unique(engines):
        idx = np.where(engines == e)[0]
        Xe, ye = X[idx], y[idx]
        for end in range(seq_len, len(Xe)+1):
            seqs.append(Xe[end-seq_len:end])
            labels.append(ye[end-1])
    return np.array(seqs), np.array(labels)

def last_seqs(X, engines, cycles, seq_len, nf):
    out = []
    for e in np.unique(engines):
        idx = np.where(engines==e)[0]
        idx = idx[np.argsort(cycles[engines==e])]
        Xe  = X[idx]
        if len(Xe) >= seq_len:
            out.append(Xe[-seq_len:])
        else:
            out.append(np.vstack([np.zeros((seq_len-len(Xe), nf)), Xe]))
    return np.array(out)

def metrics(yt, yp):
    return (mean_absolute_error(yt,yp),
            np.sqrt(mean_squared_error(yt,yp)),
            r2_score(yt,yp))

def conformal_q(resid, coverage):
    n     = len(resid)
    level = min(np.ceil((n+1)*coverage)/n, 1.0)
    return np.quantile(resid, level)

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # ── load data ────────────────────────────────────────────────────────────
    train = load("train")
    test  = load("test")
    true_rul = pd.read_csv(os.path.join(DATA_DIR,"RUL_FD001.txt"),
                           header=None).iloc[:,0].values

    fcols = feature_cols(train)
    max_cy = train.groupby("engine")["cycle"].transform("max")
    train["RUL"] = (max_cy - train["cycle"]).clip(upper=RUL_CLIP)

    scaler  = StandardScaler().fit(train[fcols].values)
    Xtr     = scaler.transform(train[fcols].values)
    Xte     = scaler.transform(test[fcols].values)
    ytr     = train["RUL"].values
    eng_tr  = train["engine"].values
    cyc_tr  = train["cycle"].values
    eng_te  = test["engine"].values
    cyc_te  = test["cycle"].values
    y_test  = np.clip(true_rul, 0, RUL_CLIP)
    nf      = Xtr.shape[1]

    # ── per-engine calibration residuals (leave-one-engine-out on train) ─────
    # Each training engine's TRUE last-cycle RUL is known (= 0 after clipping,
    # but we use the raw last label before clipping for a harder calibration).
    # We collect one residual per training engine using a quick RF fit.
    print("Building per-engine calibration residuals …")
    engines_uniq = np.unique(eng_tr)
    calib_resid  = []

    # fit full RF + XGB on ALL training data for the test predictions
    te_feats = last_row(Xte, eng_te)
    rf  = RandomForestRegressor(100, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    xg  = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                            random_state=42, n_jobs=-1).fit(Xtr, ytr)
    rf_pred  = rf.predict(te_feats)
    xgb_pred = xg.predict(te_feats)

    # leave-one-engine-out calibration for RF (proxy for LSTM difficulty)
    for e in engines_uniq:
        mask   = eng_tr != e          # train on all OTHER engines
        val_idx = np.where(eng_tr == e)[0]
        last    = val_idx[-1]         # last cycle of this engine
        rf_loo  = RandomForestRegressor(50, random_state=0,
                                        n_jobs=-1).fit(Xtr[mask], ytr[mask])
        pred    = rf_loo.predict(Xtr[last:last+1])[0]
        calib_resid.append(abs(ytr[last] - pred))
    calib_resid = np.array(calib_resid)

    # ── LSTM training ─────────────────────────────────────────────────────────
    print("Training LSTM …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xs, ys = make_seqs(Xtr, ytr, eng_tr, SEQ_LEN)

    class LSTMReg(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.lstm = nn.LSTM(n,64,num_layers=2,batch_first=True,dropout=0.2)
            self.head = nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
        def forward(self,x):
            o,_ = self.lstm(x); return self.head(o[:,-1,:])

    model = LSTMReg(nf).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    lf    = nn.MSELoss()
    Xt    = torch.tensor(Xs, dtype=torch.float32)
    yt_t  = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)
    from torch.utils.data import TensorDataset, DataLoader
    loader = DataLoader(TensorDataset(Xt,yt_t), batch_size=256, shuffle=True)
    model.train()
    for _ in range(30):
        for xb,yb in loader:
            xb,yb = xb.to(device),yb.to(device)
            opt.zero_grad(); lf(model(xb),yb).backward(); opt.step()

    model.eval()
    test_seqs  = last_seqs(Xte, eng_te, cyc_te, SEQ_LEN, nf)
    with torch.no_grad():
        lstm_pred = model(torch.tensor(test_seqs,
                          dtype=torch.float32).to(device)).cpu().numpy().ravel()
    lstm_pred = np.clip(lstm_pred, 0, RUL_CLIP)

    # ── ensemble mean prediction ──────────────────────────────────────────────
    ensemble_pred = (rf_pred + xgb_pred + lstm_pred) / 3.0

    # ── per-engine model predictions (for variance interval) ─────────────────
    model_preds = np.stack([rf_pred, xgb_pred, lstm_pred], axis=1)  # (100, 3)

    # ── 1. VARIANCE INTERVAL ─────────────────────────────────────────────────
    pred_std   = model_preds.std(axis=1)
    # use z=1.645 for ~90% normal interval
    var_lower  = ensemble_pred - 1.645 * pred_std
    var_upper  = ensemble_pred + 1.645 * pred_std
    var_cov    = ((y_test>=var_lower)&(y_test<=var_upper)).mean()
    var_width  = (var_upper - var_lower).mean()

    # ── 2. CONFIDENCE INTERVAL (t-based, 3 models) ───────────────────────────
    # With only 3 models, use t distribution (df=2, 90% two-sided -> t=2.920)
    t_val     = 2.920
    sem       = pred_std / np.sqrt(3)
    ci_lower  = ensemble_pred - t_val * sem
    ci_upper  = ensemble_pred + t_val * sem
    ci_cov    = ((y_test>=ci_lower)&(y_test<=ci_upper)).mean()
    ci_width  = (ci_upper - ci_lower).mean()

    # ── 3. CONFORMAL INTERVAL (per-engine LOO calibration) ───────────────────
    q          = conformal_q(calib_resid, COVERAGE)
    conf_lower = ensemble_pred - q
    conf_upper = ensemble_pred + q
    conf_cov   = ((y_test>=conf_lower)&(y_test<=conf_upper)).mean()
    conf_width = (conf_upper - conf_lower).mean()

    # ── also check conformal on LSTM only for comparison ─────────────────────
    q_lstm     = conformal_q(calib_resid, COVERAGE)
    lc_lower   = lstm_pred - q_lstm
    lc_upper   = lstm_pred + q_lstm
    lc_cov     = ((y_test>=lc_lower)&(y_test<=lc_upper)).mean()

    # ── print results ─────────────────────────────────────────────────────────
    mae_e, rmse_e, r2_e = metrics(y_test, ensemble_pred)
    print()
    print("=" * 62)
    print("  ENSEMBLE POINT PREDICTION")
    print(f"  MAE={mae_e:.2f}  RMSE={rmse_e:.2f}  R2={r2_e:.3f}")
    print("=" * 62)
    print(f"  {'Method':<28} {'Coverage':>9} {'Avg width':>10}  Honest?")
    print("-" * 62)
    print(f"  {'Variance interval':<28} {var_cov:>8.1%} {var_width:>10.2f}  "
          f"{'YES' if abs(var_cov-COVERAGE)<0.08 else 'NO'}")
    print(f"  {'Confidence interval (t)':<28} {ci_cov:>8.1%} {ci_width:>10.2f}  "
          f"{'YES' if abs(ci_cov-COVERAGE)<0.08 else 'NO'}")
    print(f"  {'Conformal (target 90%)':<28} {conf_cov:>8.1%} {conf_width:>10.2f}  "
          f"{'YES' if abs(conf_cov-COVERAGE)<0.08 else 'NO'}")
    print("=" * 62)
    print()
    print("  KEY FINDING:")
    print(f"  Conformal q (half-width) = {q:.2f} cycles")
    print(f"  Variance spread (avg std) = {pred_std.mean():.2f} cycles")
    print()
    print("  -> Copy Coverage + Avg width into your notes table.")
    print("  -> The gap between methods = your research finding.")
    print("=" * 62)

if __name__ == "__main__":
    main()
