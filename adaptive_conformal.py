"""
adaptive_conformal.py — Month 3: Adaptive Conformal Q

KEY IDEA:
  Current method: one fixed q for ALL test engines (q = 21 cycles for everyone).
  Problem: an engine at cycle 5 (early, easy) gets the same wide interval
           as an engine at cycle 190 (near failure, hard). Not smart.

  New method: each test engine gets its OWN q based on TWO signals:
    1. Lifecycle position: how far along is this engine? (early = narrow, late = wide)
    2. Prediction uncertainty: how much do RF/XGB/LSTM disagree? (agree = narrow, disagree = wide)

  These two signals are combined into an "uncertainty score" per engine.
  Calibration engines also get uncertainty scores.
  We match each test engine to calibration engines with SIMILAR uncertainty scores
  and use THEIR residuals to set q. This is called "mondrian conformal prediction."

WHY THIS IS NOVEL:
  No existing C-MAPSS paper uses adaptive per-engine conformal q.
  The 2023 IJPHM paper uses one fixed q for all engines.
  Our method gives tighter intervals early in life AND honest intervals near end-of-life.

Run:  python adaptive_conformal.py
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

def last_seqs(X, engines, cycles, seq_len, nf):
    out = []
    for e in np.unique(engines):
        idx  = np.where(engines == e)[0]
        idx  = idx[np.argsort(cycles[engines == e])]
        Xe   = X[idx]
        if len(Xe) >= seq_len:
            out.append(Xe[-seq_len:])
        else:
            out.append(np.vstack([np.zeros((seq_len-len(Xe), nf)), Xe]))
    return np.array(out)

def random_cutoff_seqs(X, y, engines, cycles, seq_len, nf, seed=7):
    """For calibration: cut each engine at a RANDOM mid-life point."""
    rng = np.random.RandomState(seed)
    seqs, labels, cut_cycles, eng_ids = [], [], [], []
    for e in np.unique(engines):
        idx  = np.where(engines == e)[0]
        idx  = idx[np.argsort(cycles[engines == e])]
        Xe, ye = X[idx], y[idx]
        cyc_e  = cycles[engines == e]
        cyc_e  = np.sort(cyc_e)
        n = len(Xe)
        if n < seq_len:
            continue
        end = rng.randint(seq_len, n)
        seqs.append(Xe[end-seq_len:end])
        labels.append(ye[end-1])
        cut_cycles.append(cyc_e[end-1])   # which cycle was cut
        eng_ids.append(e)
    return (np.array(seqs), np.array(labels),
            np.array(cut_cycles), np.array(eng_ids))

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

# ── uncertainty score ─────────────────────────────────────────────────────────

def uncertainty_score(rf_p, xgb_p, lstm_p, max_cycle_fraction):
    """
    Combine two signals into one uncertainty score per engine (0 to 1).

    Signal 1 — lifecycle position:
        max_cycle_fraction = (current cycle) / (max training cycle)
        Early in life (fraction near 0) = lower uncertainty
        Near end of life (fraction near 1) = higher uncertainty

    Signal 2 — model disagreement:
        std of RF/XGB/LSTM predictions
        High std = models disagree = uncertain

    Both signals are normalised to [0,1] and averaged.
    """
    # signal 1: lifecycle position (already in [0,1])
    pos_score = np.clip(max_cycle_fraction, 0, 1)

    # signal 2: model disagreement normalised by RUL_CLIP
    stack = np.stack([rf_p, xgb_p, lstm_p], axis=1)
    std   = stack.std(axis=1)
    dis_score = np.clip(std / (RUL_CLIP * 0.3), 0, 1)

    # combined score: average of both signals
    return 0.5 * pos_score + 0.5 * dis_score

def adaptive_conformal_q(cal_scores, cal_resid, test_scores,
                          coverage=0.90, k=8):
    """
    Mondrian conformal: for each test engine, find the k nearest
    calibration engines by uncertainty score and use THEIR residuals
    to compute q. This gives each test engine its own personalised q.

    Parameters:
        cal_scores  : uncertainty scores for calibration engines (n_cal,)
        cal_resid   : absolute residuals for calibration engines (n_cal,)
        test_scores : uncertainty scores for test engines (n_test,)
        k           : how many nearest calibration neighbours to use
    """
    qs = []
    for ts in test_scores:
        # find k nearest calibration engines by score distance
        dists  = np.abs(cal_scores - ts)
        nn_idx = np.argsort(dists)[:k]
        local_resid = cal_resid[nn_idx]
        # compute conformal quantile on local residuals
        n     = len(local_resid)
        level = min(np.ceil((n+1)*coverage)/n, 1.0)
        qs.append(np.quantile(local_resid, level))
    return np.array(qs)

def global_conformal_q(resid, coverage=0.90):
    n     = len(resid)
    level = min(np.ceil((n+1)*coverage)/n, 1.0)
    return float(np.quantile(resid, level))

def coverage_width(y, lo, hi):
    cov   = ((y >= lo) & (y <= hi)).mean()
    width = (hi - lo).mean()
    return cov, width

# ── main ─────────────────────────────────────────────────────────────────────

def run_fd(fd="FD001"):
    print(f"\n{'='*60}")
    print(f"  Running adaptive conformal on {fd} ...")
    print(f"{'='*60}")

    # load data
    train = load("train", fd)
    test  = load("test",  fd)
    true_rul = pd.read_csv(
        os.path.join(DATA_DIR, f"RUL_{fd}.txt"),
        header=None).iloc[:,0].values

    fc = [c for c in COLS if c.startswith("s") and c not in DROP]
    mc = train.groupby("engine")["cycle"].transform("max")
    train["RUL"] = (mc - train["cycle"]).clip(upper=RUL_CLIP)

    # engine split: 80 train / 20 calibration
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
    cyc_tr = train.loc[mask_tr, "cycle"].values
    eng_ca = train.loc[mask_ca, "engine"].values
    cyc_ca = train.loc[mask_ca, "cycle"].values
    eng_te = test["engine"].values
    cyc_te = test["cycle"].values
    y_test = np.clip(true_rul, 0, RUL_CLIP)
    nf     = Xtr.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # max cycle in training (for lifecycle fraction)
    max_tr_cycle = train.loc[mask_tr, "cycle"].max()

    # train models
    print(f"  Training RF + XGB ...")
    rf = RandomForestRegressor(100, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    xg = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                           random_state=42, n_jobs=-1).fit(Xtr, ytr)
    print(f"  Training LSTM ({device}) ...")
    Xs, ys = make_seqs(Xtr, ytr, eng_tr, SEQ_LEN)
    lstm   = train_lstm(build_lstm(nf, device), Xs, ys, device)

    # ── CALIBRATION with random cutoff ───────────────────────────────────────
    cal_seqs, cal_y, cal_cut_cyc, _ = random_cutoff_seqs(
        Xca, yca, eng_ca, cyc_ca, SEQ_LEN, nf, seed=7)
    cal_X    = cal_seqs[:,-1,:]
    rf_cal   = rf.predict(cal_X)
    xgb_cal  = xg.predict(cal_X)
    lstm_cal = predict_seqs(lstm, cal_seqs, device)
    ens_cal  = (rf_cal + xgb_cal + lstm_cal) / 3.0
    cal_resid = np.abs(cal_y - ens_cal)

    # lifecycle fraction for calibration engines
    cal_frac = np.clip(cal_cut_cyc / max_tr_cycle, 0, 1)

    # uncertainty scores for calibration engines
    cal_scores = uncertainty_score(rf_cal, xgb_cal, lstm_cal, cal_frac)

    # ── TEST predictions ─────────────────────────────────────────────────────
    te_seqs  = last_seqs(Xte, eng_te, cyc_te, SEQ_LEN, nf)
    te_X     = te_seqs[:,-1,:]
    rf_te    = rf.predict(te_X)
    xgb_te   = xg.predict(te_X)
    lstm_te  = predict_seqs(lstm, te_seqs, device)
    ens_te   = (rf_te + xgb_te + lstm_te) / 3.0

    # lifecycle fraction for test engines (last cycle / max train cycle)
    te_last_cyc = np.array([
        cyc_te[eng_te == e].max() for e in np.unique(eng_te)])
    te_frac = np.clip(te_last_cyc / max_tr_cycle, 0, 1)

    # uncertainty scores for test engines
    te_scores = uncertainty_score(rf_te, xgb_te, lstm_te, te_frac)

    # point metrics
    mae  = mean_absolute_error(y_test, ens_te)
    rmse = np.sqrt(mean_squared_error(y_test, ens_te))
    r2   = r2_score(y_test, ens_te)

    # ── METHOD 1: global random-cutoff conformal (your existing method) ───────
    q_global  = global_conformal_q(cal_resid, COVERAGE)
    lo_g, hi_g = ens_te - q_global, ens_te + q_global
    cov_g, wid_g = coverage_width(y_test, lo_g, hi_g)

    # ── METHOD 2: adaptive conformal (NEW contribution) ────────────────────
    q_adaptive = adaptive_conformal_q(
        cal_scores, cal_resid, te_scores, COVERAGE, k=8)
    lo_a, hi_a = ens_te - q_adaptive, ens_te + q_adaptive
    cov_a, wid_a = coverage_width(y_test, lo_a, hi_a)

    # ── coverage by lifecycle stage ───────────────────────────────────────────
    # split test engines into early life (<50% of max cycle) and late life (>=50%)
    early_mask = te_frac < 0.5
    late_mask  = ~early_mask

    def stage_cov(lo, hi, mask):
        if mask.sum() == 0:
            return float('nan')
        return ((y_test[mask] >= lo[mask]) &
                (y_test[mask] <= hi[mask])).mean()

    # global q by stage
    cov_g_early = stage_cov(lo_g, hi_g, early_mask)
    cov_g_late  = stage_cov(lo_g, hi_g, late_mask)

    # adaptive q by stage
    cov_a_early = stage_cov(lo_a, hi_a, early_mask)
    cov_a_late  = stage_cov(lo_a, hi_a, late_mask)

    # average q values by stage
    q_a_early = q_adaptive[early_mask].mean() if early_mask.sum() > 0 else float('nan')
    q_a_late  = q_adaptive[late_mask].mean()  if late_mask.sum()  > 0 else float('nan')

    print(f"\n  Point prediction: MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f}")
    print(f"\n  {'Method':<35}{'Coverage':>10}{'Avg width':>11}{'Honest?':>9}")
    print(f"  {'-'*65}")
    def h(c): return "YES" if abs(c-COVERAGE)<0.10 else "NO"
    print(f"  {'Global random-cutoff (existing)':<35}{cov_g:>9.1%}{wid_g:>11.2f}{h(cov_g):>9}")
    print(f"  {'Adaptive conformal (NEW)':<35}{cov_a:>9.1%}{wid_a:>11.2f}{h(cov_a):>9}")

    print(f"\n  Coverage by lifecycle stage:")
    print(f"  {'Stage':<20}{'N engines':>10}{'Global cov':>12}{'Adaptive cov':>14}{'Adapt. avg q':>14}")
    print(f"  {'-'*70}")
    print(f"  {'Early life (<50%)':<20}{early_mask.sum():>10}"
          f"{cov_g_early:>11.1%}{cov_a_early:>13.1%}{q_a_early:>14.2f}")
    print(f"  {'Late life (>=50%)':<20}{late_mask.sum():>10}"
          f"{cov_g_late:>11.1%}{cov_a_late:>13.1%}{q_a_late:>14.2f}")

    print(f"\n  Adaptive q stats:")
    print(f"    Min q : {q_adaptive.min():.2f} cycles (most confident engine)")
    print(f"    Max q : {q_adaptive.max():.2f} cycles (least confident engine)")
    print(f"    Mean q: {q_adaptive.mean():.2f} cycles")
    print(f"    Global q was: {q_global:.2f} cycles (same for everyone)")

    return {
        "fd": fd,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 3),
        "global_cov": f"{cov_g:.0%}",
        "global_wid": round(wid_g, 2),
        "adapt_cov": f"{cov_a:.0%}",
        "adapt_wid": round(wid_a, 2),
        "adapt_q_min": round(q_adaptive.min(), 2),
        "adapt_q_max": round(q_adaptive.max(), 2),
        "adapt_q_mean": round(q_adaptive.mean(), 2),
        "global_q": round(q_global, 2),
        "cov_g_early": f"{cov_g_early:.0%}",
        "cov_g_late": f"{cov_g_late:.0%}",
        "cov_a_early": f"{cov_a_early:.0%}",
        "cov_a_late": f"{cov_a_late:.0%}",
    }

def main():
    all_results = []
    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        try:
            r = run_fd(fd)
            all_results.append(r)
        except Exception as ex:
            print(f"  ERROR on {fd}: {ex}")

    # final summary table
    print(f"\n\n{'='*80}")
    print("  FINAL SUMMARY — GLOBAL vs ADAPTIVE CONFORMAL (all subsets)")
    print(f"{'='*80}")
    print(f"  {'Subset':<8}{'Global cov':>12}{'Global wid':>12}"
          f"{'Adapt cov':>11}{'Adapt wid':>11}{'q min':>8}{'q max':>8}")
    print(f"  {'-'*70}")
    for r in all_results:
        print(f"  {r['fd']:<8}{r['global_cov']:>12}{r['global_wid']:>12}"
              f"{r['adapt_cov']:>11}{r['adapt_wid']:>11}"
              f"{r['adapt_q_min']:>8}{r['adapt_q_max']:>8}")
    print(f"{'='*80}")
    print()
    print("  KEY things to look for:")
    print("  1. Adaptive width < Global width -> intervals are TIGHTER (better)")
    print("  2. Adaptive coverage still near 90% -> still HONEST")
    print("  3. q min vs q max -> shows how much q varies per engine (novel)")
    print("  4. Late-life coverage -> should improve vs global (key finding)")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
