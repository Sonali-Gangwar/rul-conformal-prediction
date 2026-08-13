"""
ablation_study.py — Month 1, Week 2

Tests four ablations on FD001 to prove each component matters:

  A) Full method (your random-cutoff conformal)        <- baseline to beat
  B) Remove MAD filtering                              <- does cleaning help?
  C) Use only LSTM (no ensemble)                       <- does ensemble help?
  D) Vary calibration size (10, 20, 30, 40 engines)   <- how many cal engines needed?

Run:  python ablation_study.py
"""

import os, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DATA_DIR = "data"
FD       = "FD001"
SEQ_LEN  = 30
RUL_CLIP = 125
COVERAGE = 0.90
COLS     = ["engine","cycle","op1","op2","op3"] + [f"s{i}" for i in range(1,22)]
DROP     = {"s1","s5","s6","s10","s16","s18","s19"}

# ── helpers ──────────────────────────────────────────────────────────────────

def load_fd(split):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{split}_{FD}.txt"),
                     sep=r"\s+", header=None).iloc[:, :26]
    df.columns = COLS
    return df

def apply_mad_filter(df, fc, threshold=3.0):
    """Remove rows where any sensor is a MAD outlier within that engine."""
    df = df.copy()
    for col in fc:
        for eng in df["engine"].unique():
            mask = df["engine"] == eng
            vals = df.loc[mask, col].values
            med  = np.median(vals)
            mad  = np.median(np.abs(vals - med))
            if mad == 0:
                continue
            outlier = np.abs(vals - med) > threshold * mad
            df.loc[mask & df[col].isin(vals[outlier]), col] = med
    return df

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
        idx = np.where(engines == e)[0]
        idx = idx[np.argsort(cycles[engines == e])]
        Xe  = X[idx]
        if len(Xe) >= seq_len:
            out.append(Xe[-seq_len:])
        else:
            out.append(np.vstack([np.zeros((seq_len-len(Xe), nf)), Xe]))
    return np.array(out)

def random_cutoff_seqs(X, y, engines, cycles, seq_len, nf, seed=7):
    rng = np.random.RandomState(seed)
    seqs, labels = [], []
    for e in np.unique(engines):
        idx = np.where(engines == e)[0]
        idx = idx[np.argsort(cycles[engines == e])]
        Xe, ye = X[idx], y[idx]
        n = len(Xe)
        if n < seq_len:
            continue
        end = rng.randint(seq_len, n)
        seqs.append(Xe[end-seq_len:end])
        labels.append(ye[end-1])
    return np.array(seqs), np.array(labels)

def build_lstm(nf, device):
    class LSTMReg(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.lstm = nn.LSTM(n, 64, num_layers=2,
                                batch_first=True, dropout=0.2)
            self.head = nn.Sequential(
                nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1))
        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(o[:,-1,:])
    return LSTMReg(nf).to(device)

def train_lstm(model, Xs, ys, device, epochs=30):
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    lf     = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(Xs, dtype=torch.float32),
                      torch.tensor(ys, dtype=torch.float32).unsqueeze(1)),
        batch_size=256, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            lf(model(xb), yb).backward()
            opt.step()
    return model

def predict_seqs(model, seqs, device):
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(seqs, dtype=torch.float32).to(device)
                  ).cpu().numpy().ravel()
    return np.clip(p, 0, RUL_CLIP)

def conf_q(resid, cov):
    n     = len(resid)
    level = min(np.ceil((n+1)*cov)/n, 1.0)
    return float(np.quantile(resid, level))

def run_experiment(label, use_mad=True, ensemble=True, n_cal=20):
    """Run one ablation experiment. Returns dict of results."""
    print(f"\n  Running: {label} ...")

    train = load_fd("train")
    test  = load_fd("test")
    true_rul = pd.read_csv(
        os.path.join(DATA_DIR, f"RUL_{FD}.txt"),
        header=None).iloc[:,0].values

    fc = [c for c in COLS if c.startswith("s") and c not in DROP]

    # optionally apply MAD filter
    if use_mad:
        train = apply_mad_filter(train, fc)

    mc = train.groupby("engine")["cycle"].transform("max")
    train["RUL"] = (mc - train["cycle"]).clip(upper=RUL_CLIP)

    # engine split
    all_eng = np.unique(train["engine"].values)
    np.random.seed(42); np.random.shuffle(all_eng)
    cal_set = set(all_eng[:n_cal])
    mask_tr = ~train["engine"].isin(cal_set)
    mask_ca =  train["engine"].isin(cal_set)

    scaler  = StandardScaler().fit(train.loc[mask_tr, fc].values)
    Xtr = scaler.transform(train.loc[mask_tr, fc].values)
    Xca = scaler.transform(train.loc[mask_ca, fc].values)
    Xte = scaler.transform(test[fc].values)

    ytr    = train.loc[mask_tr, "RUL"].values
    yca    = train.loc[mask_ca, "RUL"].values
    eng_tr = train.loc[mask_tr, "engine"].values
    eng_ca = train.loc[mask_ca, "engine"].values
    cyc_ca = train.loc[mask_ca, "cycle"].values
    eng_te = test["engine"].values
    cyc_te = test["cycle"].values
    y_test = np.clip(true_rul, 0, RUL_CLIP)
    nf     = Xtr.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train
    rf = RandomForestRegressor(100, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    xg = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                           random_state=42, n_jobs=-1).fit(Xtr, ytr)
    Xs, ys = make_seqs(Xtr, ytr, eng_tr, SEQ_LEN)
    lstm   = train_lstm(build_lstm(nf, device), Xs, ys, device)

    # test predictions
    te_seqs = last_seqs(Xte, eng_te, cyc_te, SEQ_LEN, nf)
    te_X    = te_seqs[:,-1,:]
    lstm_p  = predict_seqs(lstm, te_seqs, device)

    if ensemble:
        ens_p = (rf.predict(te_X) + xg.predict(te_X) + lstm_p) / 3.0
    else:
        ens_p = lstm_p  # LSTM only

    mae  = mean_absolute_error(y_test, ens_p)
    rmse = np.sqrt(mean_squared_error(y_test, ens_p))
    r2   = r2_score(y_test, ens_p)

    # calibration
    cal_seqs, cal_y = random_cutoff_seqs(
        Xca, yca, eng_ca, cyc_ca, SEQ_LEN, nf, seed=7)
    cal_X   = cal_seqs[:,-1,:]
    cal_lstm = predict_seqs(lstm, cal_seqs, device)

    if ensemble:
        cal_ens = (rf.predict(cal_X) + xg.predict(cal_X) + cal_lstm) / 3.0
    else:
        cal_ens = cal_lstm

    cal_resid = np.abs(cal_y - cal_ens)
    q         = conf_q(cal_resid, COVERAGE)
    lo, hi    = ens_p - q, ens_p + q
    coverage  = ((y_test >= lo) & (y_test <= hi)).mean()
    width     = (hi - lo).mean()

    return {
        "label":    label,
        "MAE":      round(mae, 2),
        "RMSE":     round(rmse, 2),
        "R2":       round(r2, 3),
        "q":        round(q, 2),
        "Coverage": f"{coverage:.0%}",
        "Width":    round(width, 2),
        "Honest":   "YES" if abs(coverage - COVERAGE) < 0.10 else "NO",
    }

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    results = []

    # A: Full method
    results.append(run_experiment(
        "A) Full method (ensemble + MAD + 20 cal)",
        use_mad=True, ensemble=True, n_cal=20))

    # B: No MAD filtering
    results.append(run_experiment(
        "B) No MAD filtering",
        use_mad=False, ensemble=True, n_cal=20))

    # C: LSTM only (no ensemble)
    results.append(run_experiment(
        "C) LSTM only (no ensemble)",
        use_mad=True, ensemble=False, n_cal=20))

    # D: Vary calibration size
    for n_cal in [10, 20, 30, 40]:
        results.append(run_experiment(
            f"D) Calibration size = {n_cal} engines",
            use_mad=True, ensemble=True, n_cal=n_cal))

    # print table
    print("\n\n" + "=" * 80)
    print("  ABLATION STUDY — FD001")
    print("=" * 80)
    print(f"  {'Experiment':<42}{'MAE':>6}{'RMSE':>7}{'R2':>7}"
          f"{'Coverage':>10}{'Width':>8}{'Honest?':>9}")
    print("-" * 80)
    for r in results:
        print(f"  {r['label']:<42}{r['MAE']:>6}{r['RMSE']:>7}"
              f"{r['R2']:>7}{r['Coverage']:>10}{r['Width']:>8}{r['Honest']:>9}")
    print("=" * 80)
    print()
    print("  What to look for:")
    print("  B vs A -> if coverage drops without MAD: MAD filtering helps")
    print("  C vs A -> if coverage drops without ensemble: ensemble helps")
    print("  D rows -> coverage should rise as calibration size increases")
    print("=" * 80)

if __name__ == "__main__":
    main()
