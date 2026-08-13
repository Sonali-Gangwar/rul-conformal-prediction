"""
adaptive_conformal_v2.py — Improved Adaptive Conformal Q

WHAT CHANGED FROM V1:
  V1 used lifecycle fraction (cycle/max_cycle) as Signal 1.
  Problem: cycle count does not capture how FAST an engine degrades.
           Two engines at cycle 100 may have very different RUL.

  V2 uses PREDICTED RUL as Signal 1 (normalised to [0,1]).
  Low predicted RUL -> near end of life -> high uncertainty -> wider q.
  High predicted RUL -> early life -> lower uncertainty -> narrower q.

  Also uses 3 signals instead of 2:
    Signal 1: normalised predicted RUL (inverted: near 0 RUL = score 1)
    Signal 2: model disagreement (std of RF/XGB/LSTM)
    Signal 3: sensor degradation trend (mean sensor change over last 5 cycles)

  This gives a richer, more informative uncertainty score per engine.

WHY THIS MATTERS FOR THE PAPER:
  The uncertainty score is now grounded in the actual degradation state,
  not just the time elapsed. This is physically meaningful and
  defensible to reviewers.

Run:  python adaptive_conformal_v2.py
"""

import os, numpy as np, pandas as pd
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

def load(split, fd="FD001"):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{split}_{fd}.txt"),
                     sep=r"\s+", header=None).iloc[:, :26]
    df.columns = COLS
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

def last_seqs_with_trend(X, engines, cycles, seq_len, nf):
    """Return last sequences AND sensor trend for each engine."""
    seqs, trends = [], []
    for e in np.unique(engines):
        idx  = np.where(engines == e)[0]
        idx  = idx[np.argsort(cycles[engines == e])]
        Xe   = X[idx]
        if len(Xe) >= seq_len:
            seq = Xe[-seq_len:]
        else:
            seq = np.vstack([np.zeros((seq_len-len(Xe), nf)), Xe])
        seqs.append(seq)
        # sensor trend: mean absolute change over last 5 cycles
        last5 = Xe[-5:] if len(Xe) >= 5 else Xe
        trend = np.abs(np.diff(last5, axis=0)).mean() if len(last5) > 1 else 0.0
        trends.append(trend)
    return np.array(seqs), np.array(trends)

def random_cutoff_seqs_with_trend(X, y, engines, cycles, seq_len, nf, seed=7):
    rng = np.random.RandomState(seed)
    seqs, labels, trends = [], [], []
    for e in np.unique(engines):
        idx  = np.where(engines == e)[0]
        idx  = idx[np.argsort(cycles[engines == e])]
        Xe, ye = X[idx], y[idx]
        n = len(Xe)
        if n < seq_len:
            continue
        end = rng.randint(seq_len, n)
        seq = Xe[end-seq_len:end]
        seqs.append(seq)
        labels.append(ye[end-1])
        # sensor trend at cutoff point
        last5 = Xe[max(0,end-5):end]
        trend = np.abs(np.diff(last5, axis=0)).mean() if len(last5) > 1 else 0.0
        trends.append(trend)
    return np.array(seqs), np.array(labels), np.array(trends)

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

def uncertainty_score_v2(rf_p, xgb_p, lstm_p, trends,
                          max_trend=None, alpha=0.4, beta=0.3, gamma=0.3):
    """
    Three-signal uncertainty score (all in [0,1], higher = more uncertain).

    Signal 1 (weight alpha): RUL proximity to failure
        = 1 - (predicted_RUL / RUL_CLIP)
        Engine predicted to fail soon (low RUL) -> score near 1 (uncertain)
        Engine healthy (high RUL) -> score near 0 (confident)

    Signal 2 (weight beta): model disagreement
        = std(RF, XGB, LSTM) / (0.3 * RUL_CLIP)
        High disagreement -> uncertain

    Signal 3 (weight gamma): sensor degradation trend
        = mean absolute sensor change over last 5 cycles (normalised)
        Rapidly changing sensors -> degrading fast -> uncertain

    Weights alpha + beta + gamma = 1.0
    """
    ens = (rf_p + xgb_p + lstm_p) / 3.0

    # signal 1: proximity to failure
    s1 = np.clip(1.0 - ens / RUL_CLIP, 0, 1)

    # signal 2: model disagreement
    stack = np.stack([rf_p, xgb_p, lstm_p], axis=1)
    std   = stack.std(axis=1)
    s2    = np.clip(std / (0.3 * RUL_CLIP), 0, 1)

    # signal 3: sensor trend (normalise by max observed trend)
    if max_trend is None or max_trend == 0:
        max_trend = trends.max() if trends.max() > 0 else 1.0
    s3 = np.clip(trends / max_trend, 0, 1)

    return alpha * s1 + beta * s2 + gamma * s3

def adaptive_conformal_q(cal_scores, cal_resid, te_scores,
                          coverage=0.90, k=8):
    """Mondrian conformal: per-engine q from k nearest calibration neighbours."""
    qs = []
    for ts in te_scores:
        dists      = np.abs(cal_scores - ts)
        nn_idx     = np.argsort(dists)[:k]
        local_res  = cal_resid[nn_idx]
        n          = len(local_res)
        level      = min(np.ceil((n+1)*coverage)/n, 1.0)
        qs.append(np.quantile(local_res, level))
    return np.array(qs)

def global_q(resid, coverage=0.90):
    n     = len(resid)
    level = min(np.ceil((n+1)*coverage)/n, 1.0)
    return float(np.quantile(resid, level))

def cov_wid(y, lo, hi):
    return ((y>=lo)&(y<=hi)).mean(), (hi-lo).mean()

def stage_coverage(y, lo, hi, mask):
    if mask.sum() == 0:
        return float('nan')
    return ((y[mask]>=lo[mask])&(y[mask]<=hi[mask])).mean()

# ── main ─────────────────────────────────────────────────────────────────────

def run_fd(fd="FD001"):
    print(f"\n{'='*62}")
    print(f"  {fd} ...")
    print(f"{'='*62}")

    train = load("train", fd)
    test  = load("test",  fd)
    true_rul = pd.read_csv(
        os.path.join(DATA_DIR, f"RUL_{fd}.txt"),
        header=None).iloc[:,0].values

    fc = [c for c in COLS if c.startswith("s") and c not in DROP]
    mc = train.groupby("engine")["cycle"].transform("max")
    train["RUL"] = (mc - train["cycle"]).clip(upper=RUL_CLIP)

    all_eng = np.unique(train["engine"].values)
    np.random.seed(42); np.random.shuffle(all_eng)
    n_cal   = max(10, int(len(all_eng)*0.2))
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

    print(f"  Training RF + XGB ...")
    rf = RandomForestRegressor(100, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    xg = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                           random_state=42, n_jobs=-1).fit(Xtr, ytr)
    print(f"  Training LSTM ({device}) ...")
    Xs, ys = make_seqs(Xtr, ytr, eng_tr, SEQ_LEN)
    lstm   = train_lstm(build_lstm(nf, device), Xs, ys, device)

    # calibration — random cutoff with trend
    cal_seqs, cal_y, cal_trends = random_cutoff_seqs_with_trend(
        Xca, yca, eng_ca, cyc_ca, SEQ_LEN, nf, seed=7)
    cal_X    = cal_seqs[:,-1,:]
    rf_cal   = rf.predict(cal_X)
    xgb_cal  = xg.predict(cal_X)
    lstm_cal = predict_seqs(lstm, cal_seqs, device)
    cal_resid = np.abs(cal_y - (rf_cal+xgb_cal+lstm_cal)/3.0)
    max_trend = cal_trends.max() if cal_trends.max() > 0 else 1.0
    cal_scores = uncertainty_score_v2(
        rf_cal, xgb_cal, lstm_cal, cal_trends, max_trend)

    # test predictions with trend
    te_seqs, te_trends = last_seqs_with_trend(
        Xte, eng_te, cyc_te, SEQ_LEN, nf)
    te_X     = te_seqs[:,-1,:]
    rf_te    = rf.predict(te_X)
    xgb_te   = xg.predict(te_X)
    lstm_te  = predict_seqs(lstm, te_seqs, device)
    ens_te   = (rf_te+xgb_te+lstm_te)/3.0
    te_scores = uncertainty_score_v2(
        rf_te, xgb_te, lstm_te, te_trends, max_trend)

    mae  = mean_absolute_error(y_test, ens_te)
    rmse = np.sqrt(mean_squared_error(y_test, ens_te))
    r2   = r2_score(y_test, ens_te)

    # global conformal
    q_g  = global_q(cal_resid, COVERAGE)
    lo_g, hi_g = ens_te-q_g, ens_te+q_g
    cov_g, wid_g = cov_wid(y_test, lo_g, hi_g)

    # adaptive conformal v2
    q_a  = adaptive_conformal_q(cal_scores, cal_resid, te_scores,
                                  COVERAGE, k=8)
    lo_a, hi_a = ens_te-q_a, ens_te+q_a
    cov_a, wid_a = cov_wid(y_test, lo_a, hi_a)

    # lifecycle stage (using predicted RUL, not cycle count)
    early = ens_te >= 50   # predicted to have 50+ cycles left = early
    late  = ~early         # predicted <50 cycles left = near failure

    print(f"\n  MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f}")
    print(f"\n  {'Method':<35}{'Coverage':>10}{'Width':>10}{'Honest?':>9}")
    print(f"  {'-'*64}")
    h = lambda c: "YES" if abs(c-COVERAGE)<0.10 else "NO"
    print(f"  {'Standard conformal (last-cycle)':<35}  [see run_all_subsets.py]")
    print(f"  {'Global random-cutoff (v5)':<35}{cov_g:>9.1%}{wid_g:>10.2f}{h(cov_g):>9}")
    print(f"  {'Adaptive conformal v2 (NEW)':<35}{cov_a:>9.1%}{wid_a:>10.2f}{h(cov_a):>9}")

    print(f"\n  Coverage by predicted RUL stage (early=RUL>=50, late=RUL<50):")
    print(f"  {'Stage':<22}{'N':>6}{'Global':>10}{'Adaptive':>11}{'Adapt q':>10}")
    print(f"  {'-'*59}")
    print(f"  {'Early (RUL >= 50)':<22}{early.sum():>6}"
          f"{stage_coverage(y_test,lo_g,hi_g,early):>9.1%}"
          f"{stage_coverage(y_test,lo_a,hi_a,early):>10.1%}"
          f"{q_a[early].mean():>10.2f}")
    print(f"  {'Late (RUL < 50)':<22}{late.sum():>6}"
          f"{stage_coverage(y_test,lo_g,hi_g,late):>9.1%}"
          f"{stage_coverage(y_test,lo_a,hi_a,late):>10.1%}"
          f"{q_a[late].mean():>10.2f}")
    print(f"\n  Adaptive q: min={q_a.min():.1f}  max={q_a.max():.1f}"
          f"  mean={q_a.mean():.1f}  global was={q_g:.1f}")

    return dict(fd=fd, MAE=round(mae,2), RMSE=round(rmse,2), R2=round(r2,3),
                cov_g=f"{cov_g:.0%}", wid_g=round(wid_g,2),
                cov_a=f"{cov_a:.0%}", wid_a=round(wid_a,2),
                q_min=round(q_a.min(),1), q_max=round(q_a.max(),1),
                q_mean=round(q_a.mean(),1), q_global=round(q_g,1),
                cov_a_late=f"{stage_coverage(y_test,lo_a,hi_a,late):.0%}",
                cov_g_late=f"{stage_coverage(y_test,lo_g,hi_g,late):.0%}")

def main():
    results = []
    for fd in ["FD001","FD002","FD003","FD004"]:
        try:
            results.append(run_fd(fd))
        except Exception as ex:
            print(f"  ERROR {fd}: {ex}")

    print(f"\n\n{'='*85}")
    print("  FINAL TABLE — ALL SUBSETS")
    print(f"{'='*85}")
    print(f"  {'FD':<6}{'MAE':>6}{'R2':>7}{'Glob-cov':>10}{'Glob-wid':>10}"
          f"{'Adap-cov':>10}{'Adap-wid':>10}{'q-min':>7}{'q-max':>7}")
    print(f"  {'-'*73}")
    for r in results:
        print(f"  {r['fd']:<6}{r['MAE']:>6}{r['R2']:>7}"
              f"{r['cov_g']:>10}{r['wid_g']:>10}"
              f"{r['cov_a']:>10}{r['wid_a']:>10}"
              f"{r['q_min']:>7}{r['q_max']:>7}")
    print(f"{'='*85}")
    print("\n  What to look for vs V1:")
    print("  - FD001 adaptive coverage should now be >= global coverage")
    print("  - Late-life coverage (RUL<50) should be high for both")
    print("  - Adaptive width should be NARROWER than global for early-life engines")
    print("  - q_min << q_max proves the method personalises per engine")
    print(f"{'='*85}")

if __name__ == "__main__":
    main()
