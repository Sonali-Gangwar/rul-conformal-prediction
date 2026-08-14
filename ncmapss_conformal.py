"""
ncmapss_conformal.py — Validate conformal RUL prediction on N-CMAPSS DS02

N-CMAPSS is more realistic than C-MAPSS:
  - Real flight conditions (climb, cruise, descent)
  - 7 different failure modes
  - Variable flight lengths per engine
  - 80 training units, 20 test units

This script runs ALL THREE conformal methods on N-CMAPSS:
  1. Standard conformal (last-cycle calibration) — should fail
  2. Global random-cutoff conformal — should fix it
  3. Adaptive Mondrian conformal — personalised per engine

If standard fails and random-cutoff fixes it here too →
you have cross-dataset validation for your paper.

Run:  python ncmapss_conformal.py
File: D:\\rul_project\\data\\data_set\\N-CMAPSS_DS02-006.h5
"""

import os
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# ── CONFIG ───────────────────────────────────────────────────────────────────
# Update this path to where your file is
H5_FILE  = r"D:\rul_project\data\data_set\N-CMAPSS_DS02-006.h5"
SEQ_LEN  = 30
RUL_CLIP = 125
COVERAGE = 0.90
CAL_FRAC = 0.20   # 20% of train units for calibration

# ── STEP 1: LOAD N-CMAPSS ────────────────────────────────────────────────────

def load_ncmapss(filepath):
    """
    Load N-CMAPSS DS02 from HDF5 file.

    N-CMAPSS structure:
      - W:  4 flight condition inputs (altitude, Mach, TRA, T2)
      - X_s: 14 sensor measurements (same sensors as C-MAPSS)
      - X_v: 14 virtual sensors
      - A:  auxiliary data (unit number, cycle, flight class, RUL)
      - Y:  RUL labels

    We use X_s (14 sensors) + W (4 conditions) = 18 features
    and Y for RUL labels, A for unit IDs.
    """
    print(f"Loading N-CMAPSS from {filepath} ...")
    print("This may take 1-2 minutes (file is ~2.3 GB) ...")

    with h5py.File(filepath, "r") as f:
        print(f"  Keys in file: {list(f.keys())}")

        # Development (training) set
        W_dev   = f["W_dev"][:]      # flight conditions (n, 4)
        Xs_dev  = f["X_s_dev"][:]    # sensor readings (n, 14)
        A_dev   = f["A_dev"][:]      # auxiliary: unit, cycle, Fc, RUL
        Y_dev   = f["Y_dev"][:]      # RUL labels (n,)

        # Test set
        W_test  = f["W_test"][:]
        Xs_test = f["X_s_test"][:]
        A_test  = f["A_test"][:]
        Y_test  = f["Y_test"][:]

    print(f"  Dev set:  {Xs_dev.shape[0]:,} rows, "
          f"{np.unique(A_dev[:,0]).shape[0]} units")
    print(f"  Test set: {Xs_test.shape[0]:,} rows, "
          f"{np.unique(A_test[:,0]).shape[0]} units")

    # Combine sensors + flight conditions as features
    X_dev  = np.hstack([Xs_dev,  W_dev])   # (n, 18)
    X_test = np.hstack([Xs_test, W_test])

    # Extract unit IDs and RUL
    units_dev  = A_dev[:, 0].astype(int)
    units_test = A_test[:, 0].astype(int)
    rul_dev    = Y_dev.ravel()
    rul_test   = Y_test.ravel()

    return X_dev, rul_dev, units_dev, X_test, rul_test, units_test

# ── STEP 2: HELPERS ──────────────────────────────────────────────────────────

def make_seqs(X, y, units, seq_len):
    """Build sliding window sequences for LSTM."""
    seqs, labels = [], []
    for u in np.unique(units):
        idx = np.where(units == u)[0]
        Xu, yu = X[idx], y[idx]
        for end in range(seq_len, len(Xu)+1):
            seqs.append(Xu[end-seq_len:end])
            labels.append(yu[end-1])
    return np.array(seqs), np.array(labels)

def last_seqs(X, units, seq_len, nf):
    """Get last-cycle sequence for each unit (for test prediction)."""
    out = []
    for u in np.unique(units):
        idx = np.where(units == u)[0]
        Xu  = X[idx]
        if len(Xu) >= seq_len:
            out.append(Xu[-seq_len:])
        else:
            out.append(np.vstack([np.zeros((seq_len-len(Xu), nf)), Xu]))
    return np.array(out)

def random_cutoff_seqs(X, y, units, seq_len, nf, seed=7):
    """Random mid-life cutoff for calibration — your key contribution."""
    rng  = np.random.RandomState(seed)
    seqs, labels, trends = [], [], []
    for u in np.unique(units):
        idx = np.where(units == u)[0]
        Xu, yu = X[idx], y[idx]
        n = len(Xu)
        if n < seq_len:
            continue
        end = rng.randint(seq_len, n)
        seqs.append(Xu[end-seq_len:end])
        labels.append(yu[end-1])
        last5 = Xu[max(0,end-5):end]
        trends.append(np.abs(np.diff(last5,axis=0)).mean()
                      if len(last5)>1 else 0.0)
    return np.array(seqs), np.array(labels), np.array(trends)

def last_seqs_with_trend(X, units, seq_len, nf):
    seqs, trends = [], []
    for u in np.unique(units):
        idx = np.where(units == u)[0]
        Xu  = X[idx]
        seq = Xu[-seq_len:] if len(Xu)>=seq_len else \
              np.vstack([np.zeros((seq_len-len(Xu),nf)),Xu])
        seqs.append(seq)
        last5 = Xu[-5:] if len(Xu)>=5 else Xu
        trends.append(np.abs(np.diff(last5,axis=0)).mean()
                      if len(last5)>1 else 0.0)
    return np.array(seqs), np.array(trends)

def build_lstm(nf, device):
    class M(nn.Module):
        def __init__(self,n):
            super().__init__()
            self.lstm=nn.LSTM(n,64,num_layers=2,
                              batch_first=True,dropout=0.2)
            self.head=nn.Sequential(
                nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
        def forward(self,x):
            o,_=self.lstm(x); return self.head(o[:,-1,:])
    return M(nf).to(device)

def train_lstm(model, Xs, ys, device, epochs=20):
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    lf     = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(Xs, dtype=torch.float32),
                      torch.tensor(ys, dtype=torch.float32).unsqueeze(1)),
        batch_size=512, shuffle=True)
    model.train()
    for ep in range(epochs):
        total = 0
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)
            opt.zero_grad()
            loss=lf(model(xb),yb)
            loss.backward(); opt.step()
            total+=loss.item()
        if (ep+1)%5==0:
            print(f"    Epoch {ep+1}/{epochs} loss={total/len(loader):.2f}")
    return model

def predict(model, seqs, device):
    model.eval()
    with torch.no_grad():
        p=model(torch.tensor(seqs,dtype=torch.float32).to(device)
                ).cpu().numpy().ravel()
    return np.clip(p, 0, RUL_CLIP)

def gq(resid, cov=0.90):
    n=len(resid)
    return float(np.quantile(resid, min(np.ceil((n+1)*cov)/n,1.0)))

def unc_score(rf_p, xgb_p, lstm_p, trends, max_t):
    ens=(rf_p+xgb_p+lstm_p)/3.0
    s1=np.clip(1.0-ens/RUL_CLIP, 0, 1)
    s2=np.clip(np.stack([rf_p,xgb_p,lstm_p],1).std(1)/(0.3*RUL_CLIP),0,1)
    s3=np.clip(trends/max(max_t,1e-6), 0, 1)
    return 0.4*s1+0.3*s2+0.3*s3

def adaptive_q(cal_sc, cal_res, te_sc, cov=0.90, k=8):
    qs=[]
    for ts in te_sc:
        nn_idx=np.argsort(np.abs(cal_sc-ts))[:k]
        lr=cal_res[nn_idx]; n=len(lr)
        qs.append(np.quantile(lr,min(np.ceil((n+1)*cov)/n,1.0)))
    return np.array(qs)

def cw(y, lo, hi):
    return ((y>=lo)&(y<=hi)).mean(), (hi-lo).mean()

def late_cov(y, lo, hi, pred):
    """Coverage for engines with predicted RUL < 50 (late life)."""
    late = pred < 50
    if late.sum() == 0:
        return float('nan')
    return ((y[late]>=lo[late])&(y[late]<=hi[late])).mean()

# ── STEP 3: MAIN ─────────────────────────────────────────────────────────────

def main():
    # Load data
    X_dev, rul_dev, units_dev, X_test, rul_test, units_test = \
        load_ncmapss(H5_FILE)

    # Clip RUL (N-CMAPSS RUL can be very large — clip same as C-MAPSS)
    rul_dev  = np.clip(rul_dev,  0, RUL_CLIP)
    rul_test = np.clip(rul_test, 0, RUL_CLIP)

    nf = X_dev.shape[1]
    print(f"\n  Features: {nf}, Train units: {len(np.unique(units_dev))}, "
          f"Test units: {len(np.unique(units_test))}")

    # Engine split: 80% train, 20% calibration
    all_units = np.unique(units_dev)
    np.random.seed(42); np.random.shuffle(all_units)
    n_cal    = max(5, int(len(all_units)*CAL_FRAC))
    cal_set  = set(all_units[:n_cal])
    mask_tr  = ~np.isin(units_dev, list(cal_set))
    mask_ca  =  np.isin(units_dev, list(cal_set))

    print(f"  Train engines: {mask_tr.sum():,} rows | "
          f"Cal engines: {n_cal} units")

    # Scale features
    scaler = StandardScaler().fit(X_dev[mask_tr])
    Xtr = scaler.transform(X_dev[mask_tr])
    Xca = scaler.transform(X_dev[mask_ca])
    Xte = scaler.transform(X_test)

    ytr = rul_dev[mask_tr]
    yca = rul_dev[mask_ca]
    etr = units_dev[mask_tr]
    eca = units_dev[mask_ca]
    ete = units_test
    # Take last-cycle RUL for each test unit (one value per unit)
    y_test = np.array([rul_test[np.where(units_test==u)[0][-1]]
                       for u in np.unique(units_test)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Train models
    print("\n  Training RF + XGB ...")
    te_last  = last_seqs(Xte, ete, SEQ_LEN, nf)[:, -1, :]
    tr_last  = np.array([Xtr[np.where(etr==u)[0][-1]]
                         for u in np.unique(etr)])

    rf = RandomForestRegressor(100, random_state=42,
                               n_jobs=-1).fit(Xtr, ytr)
    xg = xgb.XGBRegressor(n_estimators=100, max_depth=6,
                           learning_rate=0.1, random_state=42,
                           n_jobs=-1).fit(Xtr, ytr)

    print(f"  Training LSTM ({device}) — 20 epochs ...")
    Xs, ys = make_seqs(Xtr, ytr, etr, SEQ_LEN)
    print(f"  LSTM sequences: {len(Xs):,}")
    lstm = train_lstm(build_lstm(nf, device), Xs, ys, device, epochs=20)

    # Test predictions
    print("\n  Predicting on test units ...")
    te_seqs, te_trends = last_seqs_with_trend(Xte, ete, SEQ_LEN, nf)
    te_X = te_seqs[:, -1, :]

    rf_p   = rf.predict(te_X)
    xgb_p  = xg.predict(te_X)
    lstm_p = predict(lstm, te_seqs, device)
    ens_p  = (rf_p + xgb_p + lstm_p) / 3.0

    mae  = mean_absolute_error(y_test, ens_p)
    rmse = np.sqrt(mean_squared_error(y_test, ens_p))
    r2   = r2_score(y_test, ens_p)

    # ── CALIBRATION 1: last-cycle (standard) ─────────────────────────────────
    ca_last_X = np.array([Xca[np.where(eca==u)[0][-1]]
                          for u in np.unique(eca)])
    ca_last_y = np.array([yca[np.where(eca==u)[0][-1]]
                          for u in np.unique(eca)])
    ca_last_seq = last_seqs(Xca, eca, SEQ_LEN, nf)
    ca_ens_last = (rf.predict(ca_last_X) +
                   xg.predict(ca_last_X) +
                   predict(lstm, ca_last_seq, device)) / 3.0
    resid_std   = np.abs(ca_last_y - ca_ens_last)
    q_std       = gq(resid_std)

    lo_std, hi_std = ens_p-q_std, ens_p+q_std
    cov_std, wid_std = cw(y_test, lo_std, hi_std)
    lc_std = late_cov(y_test, lo_std, hi_std, ens_p)

    # ── CALIBRATION 2: random-cutoff (your fix) ───────────────────────────────
    ca_seqs, ca_y, ca_tr = random_cutoff_seqs(Xca, yca, eca, SEQ_LEN, nf)
    ca_X   = ca_seqs[:, -1, :]
    ca_ens = (rf.predict(ca_X) +
              xg.predict(ca_X) +
              predict(lstm, ca_seqs, device)) / 3.0
    resid_rc = np.abs(ca_y - ca_ens)
    q_rc     = gq(resid_rc)

    lo_rc, hi_rc = ens_p-q_rc, ens_p+q_rc
    cov_rc, wid_rc = cw(y_test, lo_rc, hi_rc)
    lc_rc = late_cov(y_test, lo_rc, hi_rc, ens_p)

    # ── CALIBRATION 3: adaptive Mondrian ─────────────────────────────────────
    mt       = ca_tr.max() if ca_tr.max()>0 else 1.0
    ca_sc    = unc_score(rf.predict(ca_X), xg.predict(ca_X),
                         predict(lstm, ca_seqs, device), ca_tr, mt)
    te_sc    = unc_score(rf_p, xgb_p, lstm_p, te_trends, mt)
    q_ad     = adaptive_q(ca_sc, resid_rc, te_sc)

    lo_ad, hi_ad = ens_p-q_ad, ens_p+q_ad
    cov_ad, wid_ad = cw(y_test, lo_ad, hi_ad)
    lc_ad = late_cov(y_test, lo_ad, hi_ad, ens_p)

    # ── PRINT RESULTS ─────────────────────────────────────────────────────────
    h = lambda c: "YES ✓" if abs(c-COVERAGE)<0.10 else "NO ✗"

    print(f"\n{'='*65}")
    print(f"  N-CMAPSS DS02 RESULTS")
    print(f"{'='*65}")
    print(f"  Point prediction:  MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}")
    print(f"  Calibration set:   {n_cal} units  |  "
          f"Test set: {len(np.unique(ete))} units")
    print(f"\n  {'Method':<30}{'Coverage':>10}{'Width':>9}"
          f"{'Late-cov':>11}{'q':>8}  Honest?")
    print(f"  {'-'*70}")
    print(f"  {'Standard conformal':<30}{cov_std:>9.1%}{wid_std:>9.2f}"
          f"{lc_std:>10.1%}{q_std:>8.2f}  {h(cov_std)}")
    print(f"  {'Global random-cutoff':<30}{cov_rc:>9.1%}{wid_rc:>9.2f}"
          f"{lc_rc:>10.1%}{q_rc:>8.2f}  {h(cov_rc)}")
    print(f"  {'Adaptive Mondrian':<30}{cov_ad:>9.1%}{wid_ad:>9.2f}"
          f"{lc_ad:>10.1%}  {q_ad.min():.1f}-{q_ad.max():.1f}  {h(cov_ad)}")
    print(f"{'='*65}")
    print(f"\n  KEY: Does the calibration-distribution mismatch")
    print(f"  appear on N-CMAPSS as it did on C-MAPSS?")
    print(f"  Standard cov={cov_std:.0%} vs target 90% → "
          f"{'YES - mismatch confirmed!' if cov_std<0.75 else 'Weak mismatch'}")
    print(f"  Random-cutoff fixes it: {cov_rc:.0%} coverage → "
          f"{'YES - fix works!' if cov_rc>=0.80 else 'Needs investigation'}")
    print(f"{'='*65}")
    print(f"\n  → Copy these numbers into your paper Table 3 as")
    print(f"    a new row: 'N-CMAPSS DS02'")
    print(f"  → If pattern matches C-MAPSS: cross-dataset validation done.")

if __name__ == "__main__":
    main()
