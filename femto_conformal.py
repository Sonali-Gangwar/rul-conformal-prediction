"""
femto_conformal.py — Validate conformal RUL on FEMTO Bearing Dataset

FEMTO structure:
  Learning_set/  — 6 bearings run to FAILURE (train)
    Bearing1_1/, Bearing1_2/, Bearing2_1/, Bearing2_2/, Bearing3_1/, Bearing3_2/
  Test_set/      — 11 bearings truncated MID-LIFE (test, like C-MAPSS test)
    Bearing1_3 ... Bearing3_3
  Full_Test_Set/ — same 11 bearings run to failure (for true RUL labels)

Each bearing folder contains CSV files: acc_00001.csv, acc_00002.csv ...
Each CSV: timestamp, hour, min, sec, microsec, horiz_acc, vert_acc (7 columns)

HEALTH INDICATOR:
  We extract RMS (Root Mean Square) of vibration per time window.
  RMS = sqrt(mean(acc^2)) — rises as bearing degrades.
  This converts raw vibration into a degradation signal like C-MAPSS sensors.

RUL LABEL:
  For training bearings: RUL = total_windows - current_window
  For test bearings: RUL = computed from Full_Test_Set total length minus Test_set length

Run:  python femto_conformal.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR     = r"D:\rul_project\data"
LEARN_DIR    = os.path.join(BASE_DIR, "Learning_set")
TEST_DIR     = os.path.join(BASE_DIR, "Test_set")
FULL_DIR     = os.path.join(BASE_DIR, "Full_Test_Set")

SEQ_LEN      = 10    # use last 10 time windows as input sequence
RUL_CLIP     = 125
COVERAGE     = 0.90

# ── STEP 1: EXTRACT RMS HEALTH INDICATOR ─────────────────────────────────────

def extract_rms(bearing_folder):
    """
    Read all acc_*.csv files in a bearing folder.
    Extract RMS of horizontal + vertical acceleration per file.
    Returns array of shape (n_windows, 2) — [rms_horiz, rms_vert]
    """
    files = sorted([f for f in os.listdir(bearing_folder)
                    if f.startswith("acc_") and f.endswith(".csv")])
    if len(files) == 0:
        return None

    rms_list = []
    for f in files:
        path = os.path.join(bearing_folder, f)
        try:
            df = pd.read_csv(path, header=None)
            # columns: timestamp, hour, min, sec, microsec, horiz, vert
            horiz = df.iloc[:, 4].values.astype(float)
            vert  = df.iloc[:, 5].values.astype(float)
            rms_h = np.sqrt(np.mean(horiz**2))
            rms_v = np.sqrt(np.mean(vert**2))
            rms_list.append([rms_h, rms_v])
        except Exception:
            continue

    if len(rms_list) == 0:
        return None
    return np.array(rms_list, dtype=np.float32)

def load_all_bearings(folder, clip_rul=True):
    """Load all bearings from a folder. Returns list of (rms_array, rul_array)."""
    bearings = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isdir(path):
            continue
        rms = extract_rms(path)
        if rms is None or len(rms) < SEQ_LEN + 1:
            print(f"    Skipping {name} (too short or empty)")
            continue
        n = len(rms)
        rul = np.arange(n-1, -1, -1, dtype=np.float32)  # n-1, n-2, ... 0
        if clip_rul:
            rul = np.clip(rul, 0, RUL_CLIP)
        bearings.append((name, rms, rul))
    return bearings

# ── STEP 2: BUILD SEQUENCES ───────────────────────────────────────────────────

def make_seqs(bearings, seq_len):
    """Build sliding window sequences from bearing data."""
    seqs, labels = [], []
    for name, rms, rul in bearings:
        for end in range(seq_len, len(rms)+1):
            seqs.append(rms[end-seq_len:end])
            labels.append(rul[end-1])
    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)

def random_cutoff_seqs(bearings, seq_len, n_per=20, seed=7):
    """Random mid-life cutoff per bearing — your key calibration fix."""
    rng = np.random.RandomState(seed)
    seqs, labels, trends = [], [], []
    for name, rms, rul in bearings:
        n = len(rms)
        if n < seq_len + 1:
            continue
        cutoffs = rng.choice(np.arange(seq_len, n),
                             size=min(n_per, n-seq_len),
                             replace=False)
        for end in cutoffs:
            seqs.append(rms[end-seq_len:end])
            labels.append(rul[end-1])
            last5 = rms[max(0,end-5):end]
            trends.append(np.abs(np.diff(last5,axis=0)).mean()
                          if len(last5)>1 else 0.0)
    return (np.array(seqs,dtype=np.float32),
            np.array(labels,dtype=np.float32),
            np.array(trends,dtype=np.float32))

def last_cutoff_seqs(bearings, seq_len):
    """Last-cycle calibration — standard approach."""
    seqs, labels = [], []
    for name, rms, rul in bearings:
        if len(rms) < seq_len:
            continue
        seqs.append(rms[-seq_len:])
        labels.append(rul[-1])
    return (np.array(seqs,dtype=np.float32),
            np.array(labels,dtype=np.float32))

def eval_seqs(bearings, seq_len, n_per=30, seed=99):
    """Multiple evaluation points per test bearing."""
    rng = np.random.RandomState(seed)
    seqs, labels, trends = [], [], []
    for name, rms, rul in bearings:
        n = len(rms)
        if n < seq_len:
            continue
        cutoffs = rng.choice(np.arange(seq_len, n),
                             size=min(n_per, n-seq_len),
                             replace=False)
        for end in cutoffs:
            seqs.append(rms[end-seq_len:end])
            labels.append(rul[end-1])
            last5 = rms[max(0,end-5):end]
            trends.append(np.abs(np.diff(last5,axis=0)).mean()
                          if len(last5)>1 else 0.0)
    return (np.array(seqs,dtype=np.float32),
            np.array(labels,dtype=np.float32),
            np.array(trends,dtype=np.float32))

# ── STEP 3: MODEL ─────────────────────────────────────────────────────────────

def build_lstm(nf, device):
    class M(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.lstm = nn.LSTM(n, 32, num_layers=2,
                                batch_first=True, dropout=0.1)
            self.head = nn.Sequential(
                nn.Linear(32,16), nn.ReLU(), nn.Linear(16,1))
        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(o[:,-1,:])
    return M(nf).to(device)

def train_lstm(model, Xs, ys, device, epochs=30):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lf  = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(Xs), torch.tensor(ys).unsqueeze(1)),
        batch_size=64, shuffle=True)
    model.train()
    for ep in range(epochs):
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lf(model(xb), yb)
            loss.backward(); opt.step()
            total += loss.item()
        if (ep+1) % 10 == 0:
            print(f"    Epoch {ep+1}/{epochs} loss={total/len(loader):.2f}")
    return model

def predict(model, seqs, device):
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(seqs).to(device)).cpu().numpy().ravel()
    return np.clip(p, 0, RUL_CLIP)

# ── STEP 4: CONFORMAL ─────────────────────────────────────────────────────────

def gq(resid, cov=0.90):
    n = len(resid)
    return float(np.quantile(resid, min(np.ceil((n+1)*cov)/n, 1.0)))

def unc_score(rf_p, xgb_p, lstm_p, trends, max_t):
    ens = (rf_p+xgb_p+lstm_p)/3.0
    s1  = np.clip(1.0-ens/RUL_CLIP, 0, 1)
    s2  = np.clip(np.stack([rf_p,xgb_p,lstm_p],1).std(1)/(0.3*RUL_CLIP),0,1)
    s3  = np.clip(trends/max(max_t,1e-6), 0, 1)
    return 0.4*s1+0.3*s2+0.3*s3

def adaptive_q(cal_sc, cal_res, te_sc, cov=0.90, k=8):
    qs = []
    for ts in te_sc:
        nn_idx = np.argsort(np.abs(cal_sc-ts))[:k]
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

# ── STEP 5: COMPUTE TRUE RUL FOR TEST BEARINGS ────────────────────────────────

def compute_test_rul(test_dir, full_dir):
    """
    Test bearings are truncated. True RUL = full_length - test_length.
    We compute this per bearing and assign RUL labels to test sequences.
    """
    test_bearings_raw = []
    for name in sorted(os.listdir(test_dir)):
        tpath = os.path.join(test_dir, name)
        fpath = os.path.join(full_dir, name)
        if not os.path.isdir(tpath):
            continue

        test_rms = extract_rms(tpath)
        if test_rms is None:
            continue

        # Full run length
        if os.path.isdir(fpath):
            full_rms = extract_rms(fpath)
            full_len = len(full_rms) if full_rms is not None else len(test_rms)
        else:
            full_len = len(test_rms)  # fallback

        test_len = len(test_rms)
        remaining = max(0, full_len - test_len)

        # RUL at each test window = remaining + (test_len - current_window)
        rul = np.array([remaining + (test_len - i - 1)
                        for i in range(test_len)], dtype=np.float32)
        rul = np.clip(rul, 0, RUL_CLIP)
        test_bearings_raw.append((name, test_rms, rul))
        print(f"    {name}: test_len={test_len}, full_len={full_len}, "
              f"remaining={remaining} cycles")

    return test_bearings_raw

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load training bearings (run to failure)
    print("\nLoading Learning_set (training bearings) ...")
    train_bearings = load_all_bearings(LEARN_DIR)
    print(f"  Loaded {len(train_bearings)} training bearings")
    for name, rms, rul in train_bearings:
        print(f"    {name}: {len(rms)} windows, max_rul={rul.max():.0f}")

    # Split: use 4 for training, 2 for calibration
    np.random.seed(42)
    idx = list(range(len(train_bearings)))
    np.random.shuffle(idx)
    cal_idx  = idx[:2]
    tr_idx   = idx[2:]
    tr_bear  = [train_bearings[i] for i in tr_idx]
    cal_bear = [train_bearings[i] for i in cal_idx]
    print(f"\n  Train bearings: {[b[0] for b in tr_bear]}")
    print(f"  Cal bearings:   {[b[0] for b in cal_bear]}")

    # Build training sequences
    Xs, ys = make_seqs(tr_bear, SEQ_LEN)
    nf = Xs.shape[2]
    print(f"\n  Training sequences: {len(Xs)}, features: {nf}")

    # Train models
    tr_feat = Xs[:,-1,:]  # last frame for RF/XGB
    print("\nTraining RF + XGB ...")
    rf = RandomForestRegressor(50, random_state=42, n_jobs=-1).fit(tr_feat, ys)
    xg = xgb.XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1,
                           random_state=42, n_jobs=-1).fit(tr_feat, ys)

    print(f"Training LSTM ({device}) ...")
    lstm = train_lstm(build_lstm(nf, device), Xs, ys, device, epochs=30)

    # Calibration
    print("\nBuilding calibration sets ...")

    # Standard: last cycle
    std_seqs, std_y = last_cutoff_seqs(cal_bear, SEQ_LEN)
    std_X = std_seqs[:,-1,:]
    std_ens = (rf.predict(std_X)+xg.predict(std_X)+
               predict(lstm,std_seqs,device))/3.0
    resid_std = np.abs(std_y - std_ens)
    q_std = gq(resid_std)
    print(f"  Standard cal: {len(resid_std)} pts, q={q_std:.2f}")

    # Random-cutoff: multiple points
    rc_seqs, rc_y, rc_tr = random_cutoff_seqs(cal_bear, SEQ_LEN, n_per=20)
    rc_X   = rc_seqs[:,-1,:]
    rc_ens = (rf.predict(rc_X)+xg.predict(rc_X)+
              predict(lstm,rc_seqs,device))/3.0
    resid_rc = np.abs(rc_y - rc_ens)
    q_rc = gq(resid_rc)
    mt   = rc_tr.max() if rc_tr.max()>0 else 1.0
    cal_sc = unc_score(rf.predict(rc_X),xg.predict(rc_X),
                       predict(lstm,rc_seqs,device),rc_tr,mt)
    print(f"  Random-cutoff cal: {len(resid_rc)} pts, q={q_rc:.2f}")

    # Load test bearings with true RUL
    print("\nLoading Test_set with true RUL from Full_Test_Set ...")
    test_bearings = compute_test_rul(TEST_DIR, FULL_DIR)
    print(f"  Loaded {len(test_bearings)} test bearings")

    # Build test evaluation points
    print(f"\nBuilding test evaluation points (30 per bearing) ...")
    te_seqs, te_y, te_tr = eval_seqs(test_bearings, SEQ_LEN, n_per=30)
    te_X = te_seqs[:,-1,:]
    print(f"  Total test evaluation points: {len(te_y)}")

    rf_p   = rf.predict(te_X)
    xgb_p  = xg.predict(te_X)
    lstm_p = predict(lstm, te_seqs, device)
    ens_p  = (rf_p+xgb_p+lstm_p)/3.0

    mae  = mean_absolute_error(te_y, ens_p)
    rmse = np.sqrt(mean_squared_error(te_y, ens_p))
    r2   = r2_score(te_y, ens_p)
    te_sc = unc_score(rf_p,xgb_p,lstm_p,te_tr,mt)

    # Three conformal methods
    lo_std,hi_std = ens_p-q_std, ens_p+q_std
    cov_std,wid_std = cw(te_y,lo_std,hi_std)
    lc_std = late_cov(te_y,lo_std,hi_std,ens_p)

    lo_rc,hi_rc = ens_p-q_rc, ens_p+q_rc
    cov_rc,wid_rc = cw(te_y,lo_rc,hi_rc)
    lc_rc = late_cov(te_y,lo_rc,hi_rc,ens_p)

    q_ad = adaptive_q(cal_sc,resid_rc,te_sc)
    lo_ad,hi_ad = ens_p-q_ad, ens_p+q_ad
    cov_ad,wid_ad = cw(te_y,lo_ad,hi_ad)
    lc_ad = late_cov(te_y,lo_ad,hi_ad,ens_p)

    h = lambda c: "YES ✓" if abs(c-COVERAGE)<0.10 else "NO ✗"

    print(f"\n{'='*65}")
    print(f"  FEMTO BEARING RESULTS  ({len(te_y)} evaluation points)")
    print(f"{'='*65}")
    print(f"  Point: MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}")
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
    print(f"{'='*65}")
    print(f"\n  q ratio: standard={q_std:.2f} → random-cutoff={q_rc:.2f} "
          f"({q_rc/max(q_std,0.01):.1f}x)")
    print(f"\n  DOES MISMATCH APPEAR ON FEMTO BEARING?")
    if cov_std < 0.80:
        print(f"  Standard={cov_std:.0%} vs 90% → YES - mismatch on different machine type!")
    else:
        print(f"  Standard={cov_std:.0%} — weaker mismatch (small dataset)")
    print(f"{'='*65}")
    print(f"\n  → Add row 'FEMTO Bearing' to paper Table 3")
    print(f"  → 3-dataset validation: C-MAPSS + N-CMAPSS + FEMTO = complete")

if __name__ == "__main__":
    main()
