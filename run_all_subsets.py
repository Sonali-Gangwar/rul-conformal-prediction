"""
run_all_subsets.py — Month 1, Week 1

Runs baseline (RF + XGB + LSTM) AND random-cutoff conformal
on ALL FOUR FD subsets (FD001, FD002, FD003, FD004) automatically.

Prints one clean comparison table at the end.

Run:  python run_all_subsets.py
"""

import os, time, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DATA_DIR = "data"
SEQ_LEN  = 30
RUL_CLIP = 125
COVERAGE = 0.90
COLS     = ["engine","cycle","op1","op2","op3"] + [f"s{i}" for i in range(1,22)]
DROP     = {"s1","s5","s6","s10","s16","s18","s19"}

# ── helpers ──────────────────────────────────────────────────────────────────

def load(split, fd):
    path = os.path.join(DATA_DIR, f"{split}_{fd}.txt")
    df = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, :26]
    df.columns = COLS
    return df

def fcols(df):
    return [c for c in COLS if c.startswith("s") and c not in DROP]

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
                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(o[:, -1, :])
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

def run_fd(fd):
    print(f"\n{'='*55}")
    print(f"  Running {fd} ...")
    print(f"{'='*55}")

    # load
    train = load("train", fd)
    test  = load("test",  fd)
    true_rul = pd.read_csv(
        os.path.join(DATA_DIR, f"RUL_{fd}.txt"),
        header=None).iloc[:, 0].values

    fc = fcols(train)
    mc = train.groupby("engine")["cycle"].transform("max")
    train["RUL"] = (mc - train["cycle"]).clip(upper=RUL_CLIP)

    # engine split: 80% train, 20% calibration
    all_eng = np.unique(train["engine"].values)
    np.random.seed(42); np.random.shuffle(all_eng)
    n_cal   = max(10, int(len(all_eng) * 0.2))
    cal_set = set(all_eng[:n_cal])
    mask_tr = ~train["engine"].isin(cal_set)
    mask_ca =  train["engine"].isin(cal_set)

    scaler = StandardScaler().fit(train.loc[mask_tr, fc].values)
    Xtr = scaler.transform(train.loc[mask_tr, fc].values)
    Xca = scaler.transform(train.loc[mask_ca, fc].values)
    Xte = scaler.transform(test[fc].values)

    ytr     = train.loc[mask_tr, "RUL"].values
    yca     = train.loc[mask_ca, "RUL"].values
    eng_tr  = train.loc[mask_tr, "engine"].values
    eng_ca  = train.loc[mask_ca, "engine"].values
    cyc_ca  = train.loc[mask_ca, "cycle"].values
    eng_te  = test["engine"].values
    cyc_te  = test["cycle"].values
    y_test  = np.clip(true_rul, 0, RUL_CLIP)
    nf      = Xtr.shape[1]
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train models
    print(f"  Training RF + XGB ...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42,
                               n_jobs=-1).fit(Xtr, ytr)
    xg = xgb.XGBRegressor(n_estimators=100, max_depth=6,
                           learning_rate=0.1, random_state=42,
                           n_jobs=-1).fit(Xtr, ytr)

    print(f"  Training LSTM ({device}) ...")
    t0   = time.time()
    Xs, ys = make_seqs(Xtr, ytr, eng_tr, SEQ_LEN)
    lstm = train_lstm(build_lstm(nf, device), Xs, ys, device)
    lstm_time = time.time() - t0

    # test predictions
    te_seqs = last_seqs(Xte, eng_te, cyc_te, SEQ_LEN, nf)
    te_X    = te_seqs[:, -1, :]
    rf_p    = rf.predict(te_X)
    xgb_p   = xg.predict(te_X)
    lstm_p  = predict_seqs(lstm, te_seqs, device)
    ens_p   = (rf_p + xgb_p + lstm_p) / 3.0

    mae  = mean_absolute_error(y_test, ens_p)
    rmse = np.sqrt(mean_squared_error(y_test, ens_p))
    r2   = r2_score(y_test, ens_p)

    # calibration — random cutoff
    print(f"  Computing calibration residuals ...")
    cal_seqs, cal_y = random_cutoff_seqs(
        Xca, yca, eng_ca, cyc_ca, SEQ_LEN, nf, seed=7)
    cal_X    = cal_seqs[:, -1, :]
    cal_ens  = (rf.predict(cal_X) + xg.predict(cal_X) +
                predict_seqs(lstm, cal_seqs, device)) / 3.0
    cal_resid = np.abs(cal_y - cal_ens)
    q         = conf_q(cal_resid, COVERAGE)

    # standard conformal (last cycle)
    last_X  = np.array([Xca[np.where(eng_ca==e)[0][-1]]
                        for e in np.unique(eng_ca)])
    last_y  = np.array([yca[np.where(eng_ca==e)[0][-1]]
                        for e in np.unique(eng_ca)])
    last_seq = last_seqs(Xca, eng_ca, cyc_ca, SEQ_LEN, nf)
    last_ens = (rf.predict(last_X) + xg.predict(last_X) +
                predict_seqs(lstm, last_seq, device)) / 3.0
    last_resid = np.abs(last_y - last_ens)
    q_std      = conf_q(last_resid, COVERAGE)

    # coverage
    def cov(pred, q_val):
        lo, hi = pred - q_val, pred + q_val
        return ((y_test >= lo) & (y_test <= hi)).mean()

    cov_std  = cov(ens_p, q_std)
    cov_rand = cov(ens_p, q)

    print(f"  {fd} done. MAE={mae:.2f} RMSE={rmse:.2f} "
          f"R2={r2:.3f} | std_cov={cov_std:.0%} rand_cov={cov_rand:.0%}")

    return {
        "fd": fd,
        "engines_train": len(np.unique(eng_tr)),
        "engines_cal": n_cal,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 3),
        "LSTM_time_s": round(lstm_time, 1),
        "std_conf_q": round(q_std, 2),
        "std_coverage": f"{cov_std:.0%}",
        "rand_conf_q": round(q, 2),
        "rand_coverage": f"{cov_rand:.0%}",
        "honest": "YES" if abs(cov_rand - COVERAGE) < 0.10 else "NO",
    }

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    results = []
    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        try:
            results.append(run_fd(fd))
        except Exception as ex:
            print(f"  ERROR on {fd}: {ex}")
            results.append({"fd": fd, "MAE": "ERROR"})

    # print final table
    print("\n\n" + "=" * 90)
    print("  FULL COMPARISON TABLE — ALL FOUR FD SUBSETS")
    print("=" * 90)
    hdr = f"{'Subset':<8}{'MAE':>7}{'RMSE':>7}{'R2':>7}{'Std-Cov':>10}{'Rand-Cov':>11}{'Honest?':>9}{'Std-q':>8}{'Rand-q':>9}"
    print(hdr)
    print("-" * 90)
    for r in results:
        if r.get("MAE") == "ERROR":
            print(f"  {r['fd']:<8}  ERROR")
            continue
        print(f"  {r['fd']:<8}"
              f"{r['MAE']:>7}"
              f"{r['RMSE']:>7}"
              f"{r['R2']:>7}"
              f"{r['std_coverage']:>10}"
              f"{r['rand_coverage']:>11}"
              f"{r['honest']:>9}"
              f"{r['std_conf_q']:>8}"
              f"{r['rand_conf_q']:>9}")
    print("=" * 90)
    print()
    print("  Std-Cov  = standard conformal (last-cycle calibration)")
    print("  Rand-Cov = random-cutoff conformal (YOUR METHOD)")
    print("  Honest   = within 10% of 90% target")
    print()
    print("  -> Copy this table into your paper (Table 1).")
    print("  -> If Rand-Cov is near 90% on all subsets = strong result.")
    print("  -> If Std-Cov is consistently low = proves the gap exists.")
    print("=" * 90)

if __name__ == "__main__":
    main()
