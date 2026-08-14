"""
ncmapss_conformal_v2.py — N-CMAPSS DS03 with multi-point evaluation

KEY INSIGHT:
  N-CMAPSS has very few engines (9 train, 6 test) but each engine
  has thousands of cycles. Instead of evaluating only at the LAST
  cycle of each test engine (giving only 6 data points), we evaluate
  at MANY random cutoff points across each test engine's lifecycle.
  This gives hundreds of test evaluation points — meaningful statistics.

  This also makes the evaluation MORE realistic — in real deployment,
  you predict RUL at any point during operation, not just at the end.

Run:  python ncmapss_conformal_v2.py
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

# ── CONFIG ────────────────────────────────────────────────────────────────────
H5_FILE  = r"D:\rul_project\data\data_set\N-CMAPSS_DS03-012.h5"
SEQ_LEN  = 50    # N-CMAPSS has longer sequences — use 50 cycles
RUL_CLIP = 125
COVERAGE = 0.90
N_EVAL   = 50    # evaluate at 50 random points per test engine

# ── LOAD ──────────────────────────────────────────────────────────────────────
def load_ncmapss(filepath):
    print(f"Loading {os.path.basename(filepath)} ...")
    with h5py.File(filepath, "r") as f:
        W_dev   = f["W_dev"][:]
        Xs_dev  = f["X_s_dev"][:]
        A_dev   = f["A_dev"][:]
        Y_dev   = f["Y_dev"][:]
        W_test  = f["W_test"][:]
        Xs_test = f["X_s_test"][:]
        A_test  = f["A_test"][:]
        Y_test  = f["Y_test"][:]

    X_dev  = np.hstack([Xs_dev,  W_dev]).astype(np.float32)
    X_test = np.hstack([Xs_test, W_test]).astype(np.float32)
    units_dev  = A_dev[:, 0].astype(int)
    units_test = A_test[:, 0].astype(int)
    rul_dev    = np.clip(Y_dev.ravel(), 0, RUL_CLIP).astype(np.float32)
    rul_test   = np.clip(Y_test.ravel(), 0, RUL_CLIP).astype(np.float32)

    print(f"  Train units: {len(np.unique(units_dev))}  "
          f"Test units: {len(np.unique(units_test))}")
    return X_dev, rul_dev, units_dev, X_test, rul_test, units_test

# ── SEQUENCE HELPERS ──────────────────────────────────────────────────────────
def make_seqs(X, y, units, seq_len, stride=10):
    """Build sequences with stride to reduce memory on large dataset."""
    seqs, labels = [], []
    for u in np.unique(units):
        idx = np.where(units == u)[0]
        Xu, yu = X[idx], y[idx]
        for end in range(seq_len, len(Xu)+1, stride):
            seqs.append(Xu[end-seq_len:end])
            labels.append(yu[end-1])
    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)

def random_eval_seqs(X, y, units, seq_len, n_per_unit=50, seed=42):
    """
    For evaluation: sample n_per_unit random cutoff points per unit.
    This gives meaningful coverage statistics even with few engines.
    """
    rng = np.random.RandomState(seed)
    seqs, labels, trends = [], [], []
    for u in np.unique(units):
        idx = np.where(units == u)[0]
        Xu, yu = X[idx], y[idx]
        n = len(Xu)
        if n < seq_len:
            continue
        # sample random cutoff points spread across the lifecycle
        cutoffs = rng.choice(np.arange(seq_len, n), 
                             size=min(n_per_unit, n-seq_len),
                             replace=False)
        for end in cutoffs:
            seqs.append(Xu[end-seq_len:end])
            labels.append(yu[end-1])
            last5 = Xu[max(0,end-5):end]
            trends.append(np.abs(np.diff(last5,axis=0)).mean()
                          if len(last5)>1 else 0.0)
    return (np.array(seqs, dtype=np.float32),
            np.array(labels, dtype=np.float32),
            np.array(trends, dtype=np.float32))

def random_cutoff_seqs(X, y, units, seq_len, n_per_unit=30, seed=7):
    """Random cutoff calibration — your key contribution."""
    rng = np.random.RandomState(seed)
    seqs, labels, trends = [], [], []
    for u in np.unique(units):
        idx = np.where(units == u)[0]
        Xu, yu = X[idx], y[idx]
        n = len(Xu)
        if n < seq_len:
            continue
        # multiple random cutoffs per calibration engine
        cutoffs = rng.choice(np.arange(seq_len, n),
                             size=min(n_per_unit, n-seq_len),
                             replace=False)
        for end in cutoffs:
            seqs.append(Xu[end-seq_len:end])
            labels.append(yu[end-1])
            last5 = Xu[max(0,end-5):end]
            trends.append(np.abs(np.diff(last5,axis=0)).mean()
                          if len(last5)>1 else 0.0)
    return (np.array(seqs, dtype=np.float32),
            np.array(labels, dtype=np.float32),
            np.array(trends, dtype=np.float32))

def last_cutoff_seqs(X, y, units, seq_len):
    """Last-cycle calibration — standard approach (should fail)."""
    seqs, labels = [], []
    for u in np.unique(units):
        idx = np.where(units == u)[0]
        Xu, yu = X[idx], y[idx]
        if len(Xu) < seq_len:
            continue
        seqs.append(Xu[-seq_len:])
        labels.append(yu[-1])
    return (np.array(seqs, dtype=np.float32),
            np.array(labels, dtype=np.float32))

# ── MODEL ─────────────────────────────────────────────────────────────────────
def build_lstm(nf, device):
    class M(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.lstm = nn.LSTM(n, 64, num_layers=2,
                                batch_first=True, dropout=0.2)
            self.head = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(o[:, -1, :])
    return M(nf).to(device)

def train_lstm(model, Xs, ys, device, epochs=15, batch=1024):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lf  = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(Xs), torch.tensor(ys).unsqueeze(1)),
        batch_size=batch, shuffle=True)
    model.train()
    for ep in range(epochs):
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lf(model(xb), yb)
            loss.backward(); opt.step()
            total += loss.item()
        if (ep+1) % 5 == 0:
            print(f"    Epoch {ep+1}/{epochs}  loss={total/len(loader):.2f}")
    return model

def predict_batch(model, seqs, device, batch=2048):
    model.eval()
    preds = []
    for i in range(0, len(seqs), batch):
        xb = torch.tensor(seqs[i:i+batch]).to(device)
        with torch.no_grad():
            preds.append(model(xb).cpu().numpy().ravel())
    return np.clip(np.concatenate(preds), 0, RUL_CLIP)

# ── CONFORMAL ─────────────────────────────────────────────────────────────────
def gq(resid, cov=0.90):
    n = len(resid)
    return float(np.quantile(resid, min(np.ceil((n+1)*cov)/n, 1.0)))

def unc_score(rf_p, xgb_p, lstm_p, trends, max_t):
    ens = (rf_p + xgb_p + lstm_p) / 3.0
    s1  = np.clip(1.0 - ens/RUL_CLIP, 0, 1)
    s2  = np.clip(np.stack([rf_p,xgb_p,lstm_p],1).std(1)/(0.3*RUL_CLIP),0,1)
    s3  = np.clip(trends/max(max_t,1e-6), 0, 1)
    return 0.4*s1 + 0.3*s2 + 0.3*s3

def adaptive_q(cal_sc, cal_res, te_sc, cov=0.90, k=10):
    qs = []
    for ts in te_sc:
        nn_idx = np.argsort(np.abs(cal_sc - ts))[:k]
        lr = cal_res[nn_idx]; n = len(lr)
        qs.append(np.quantile(lr, min(np.ceil((n+1)*cov)/n, 1.0)))
    return np.array(qs)

def cw(y, lo, hi):
    return ((y>=lo)&(y<=hi)).mean(), (hi-lo).mean()

def late_cov(y, lo, hi, pred):
    late = pred < 50
    if late.sum() == 0:
        return float('nan')
    return ((y[late]>=lo[late])&(y[late]<=hi[late])).mean()

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    X_dev, rul_dev, units_dev, X_test, rul_test, units_test = \
        load_ncmapss(H5_FILE)

    nf = X_dev.shape[1]
    all_units = np.unique(units_dev)
    np.random.seed(42); np.random.shuffle(all_units)

    # Use 1 unit for calibration, rest for training
    # (with only 9 train units, we cannot spare too many)
    n_cal   = 2
    cal_set = set(all_units[:n_cal])
    mask_tr = ~np.isin(units_dev, list(cal_set))
    mask_ca =  np.isin(units_dev, list(cal_set))

    print(f"\n  Using {sum(mask_tr):,} rows for training, "
          f"{sum(mask_ca):,} rows for calibration")

    scaler = StandardScaler().fit(X_dev[mask_tr])
    Xtr = scaler.transform(X_dev[mask_tr])
    Xca = scaler.transform(X_dev[mask_ca])
    Xte = scaler.transform(X_test)

    ytr     = rul_dev[mask_tr]
    yca     = rul_dev[mask_ca]
    etr     = units_dev[mask_tr]
    eca     = units_dev[mask_ca]
    ete     = units_test
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Build training sequences (stride=10 to keep memory manageable)
    print("\n  Building training sequences (stride=10) ...")
    Xs, ys = make_seqs(Xtr, ytr, etr, SEQ_LEN, stride=10)
    print(f"  Training sequences: {len(Xs):,}")

    # Feature rows for RF/XGB (use last frame of each sequence)
    tr_feat = Xs[:, -1, :]

    print("\n  Training RF + XGB ...")
    rf = RandomForestRegressor(50, random_state=42, n_jobs=-1).fit(tr_feat, ys)
    xg = xgb.XGBRegressor(n_estimators=50, max_depth=6, learning_rate=0.1,
                           random_state=42, n_jobs=-1).fit(tr_feat, ys)

    print(f"  Training LSTM ({device}) ...")
    lstm = train_lstm(build_lstm(nf, device), Xs, ys, device, epochs=15)

    # ── CALIBRATION ───────────────────────────────────────────────────────────
    print("\n  Building calibration sets ...")

    # Standard: last cycle only
    std_seqs, std_y = last_cutoff_seqs(Xca, yca, eca, SEQ_LEN)
    std_X   = std_seqs[:, -1, :]
    std_ens = (rf.predict(std_X) + xg.predict(std_X) +
               predict_batch(lstm, std_seqs, device)) / 3.0
    resid_std = np.abs(std_y - std_ens)
    q_std     = gq(resid_std)
    print(f"  Standard calibration: {len(resid_std)} points, q={q_std:.2f}")

    # Random-cutoff: multiple points per engine
    rc_seqs, rc_y, rc_tr = random_cutoff_seqs(
        Xca, yca, eca, SEQ_LEN, n_per_unit=30, seed=7)
    rc_X   = rc_seqs[:, -1, :]
    rc_ens = (rf.predict(rc_X) + xg.predict(rc_X) +
              predict_batch(lstm, rc_seqs, device)) / 3.0
    resid_rc = np.abs(rc_y - rc_ens)
    q_rc     = gq(resid_rc)
    mt       = rc_tr.max() if rc_tr.max()>0 else 1.0
    cal_sc   = unc_score(rf.predict(rc_X), xg.predict(rc_X),
                         predict_batch(lstm, rc_seqs, device), rc_tr, mt)
    print(f"  Random-cutoff calibration: {len(resid_rc)} points, q={q_rc:.2f}")

    # ── TEST EVALUATION: multiple points per test engine ─────────────────────
    print(f"\n  Building test evaluation points ({N_EVAL} per engine) ...")
    te_seqs, te_y, te_tr = random_eval_seqs(
        Xte, rul_test, ete, SEQ_LEN, n_per_unit=N_EVAL, seed=99)
    te_X = te_seqs[:, -1, :]
    print(f"  Test evaluation points: {len(te_y)}")

    rf_p   = rf.predict(te_X)
    xgb_p  = xg.predict(te_X)
    lstm_p = predict_batch(lstm, te_seqs, device)
    ens_p  = (rf_p + xgb_p + lstm_p) / 3.0

    mae  = mean_absolute_error(te_y, ens_p)
    rmse = np.sqrt(mean_squared_error(te_y, ens_p))
    r2   = r2_score(te_y, ens_p)

    te_sc = unc_score(rf_p, xgb_p, lstm_p, te_tr, mt)

    # ── THREE CONFORMAL METHODS ───────────────────────────────────────────────
    lo_std, hi_std = ens_p-q_std, ens_p+q_std
    cov_std, wid_std = cw(te_y, lo_std, hi_std)
    lc_std = late_cov(te_y, lo_std, hi_std, ens_p)

    lo_rc, hi_rc = ens_p-q_rc, ens_p+q_rc
    cov_rc, wid_rc = cw(te_y, lo_rc, hi_rc)
    lc_rc = late_cov(te_y, lo_rc, hi_rc, ens_p)

    q_ad = adaptive_q(cal_sc, resid_rc, te_sc, k=10)
    lo_ad, hi_ad = ens_p-q_ad, ens_p+q_ad
    cov_ad, wid_ad = cw(te_y, lo_ad, hi_ad)
    lc_ad = late_cov(te_y, lo_ad, hi_ad, ens_p)

    # ── PRINT ─────────────────────────────────────────────────────────────────
    h = lambda c: "YES ✓" if abs(c-COVERAGE)<0.10 else "NO ✗"

    print(f"\n{'='*68}")
    print(f"  N-CMAPSS DS03 RESULTS  ({len(te_y)} evaluation points)")
    print(f"{'='*68}")
    print(f"  Point prediction: MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}")
    print(f"\n  {'Method':<28}{'Coverage':>10}{'Width':>9}"
          f"{'Late-cov':>11}{'q':>8}  Honest?")
    print(f"  {'-'*68}")
    print(f"  {'Standard conformal':<28}{cov_std:>9.1%}{wid_std:>9.2f}"
          f"{lc_std:>10.1%}{q_std:>8.2f}  {h(cov_std)}")
    print(f"  {'Global random-cutoff':<28}{cov_rc:>9.1%}{wid_rc:>9.2f}"
          f"{lc_rc:>10.1%}{q_rc:>8.2f}  {h(cov_rc)}")
    print(f"  {'Adaptive Mondrian':<28}{cov_ad:>9.1%}{wid_ad:>9.2f}"
          f"{lc_ad:>10.1%}"
          f"  {q_ad.min():.1f}-{q_ad.max():.1f}  {h(cov_ad)}")
    print(f"{'='*68}")
    print(f"\n  Calibration q:  standard={q_std:.2f}  random-cutoff={q_rc:.2f}"
          f"  ratio={q_rc/max(q_std,0.01):.1f}x")
    print(f"\n  DOES MISMATCH APPEAR ON N-CMAPSS?")
    print(f"  Standard coverage = {cov_std:.0%} vs target 90%")
    if cov_std < 0.80:
        print(f"  → YES — mismatch confirmed on real flight data!")
    else:
        print(f"  → Mismatch weaker here (fewer engines = noisier estimate)")
    print(f"{'='*68}")

if __name__ == "__main__":
    main()
