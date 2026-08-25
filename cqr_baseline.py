"""
cqr_baseline.py — Conformalized Quantile Regression baseline for paper

Compares:
  1. Standard CP (last-cycle calibration) — already in paper
  2. CQR (Conformalized Quantile Regression) — THIS SCRIPT
  3. Global random-cutoff — already in paper
  4. Adaptive Mondrian — already in paper

CQR uses MAPIE library with QuantileRegressor as base.
We use LSTM predictions as the base estimator (same as paper).

Run: python cqr_baseline.py
Output: exact values for Tables 3 and 4 in the paper.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

DATA_DIR = "data"
SEQ_LEN  = 30
RUL_CLIP = 125
COVERAGE = 0.90
COLS     = ["engine","cycle","op1","op2","op3"] + [f"s{i}" for i in range(1,22)]
DROP     = {"s1","s5","s6","s10","s16","s18","s19"}
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── DATA ─────────────────────────────────────────────────────────────────────

def load_fd(split, fd):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{split}_{fd}.txt"),
                     sep=r"\s+", header=None).iloc[:, :26]
    df.columns = COLS
    return df

def prepare(fd):
    train = load_fd("train", fd)
    test  = load_fd("test",  fd)
    rul_t = pd.read_csv(os.path.join(DATA_DIR, f"RUL_{fd}.txt"),
                        header=None).iloc[:, 0].values
    fc = [c for c in COLS if c.startswith("s") and c not in DROP]
    mc = train.groupby("engine")["cycle"].transform("max")
    train["RUL"] = (mc - train["cycle"]).clip(upper=RUL_CLIP)
    all_e = np.unique(train["engine"].values)
    np.random.seed(42); np.random.shuffle(all_e)
    n_cal   = max(5, int(len(all_e) * 0.20))
    cal_set = set(all_e[:n_cal])
    mtr = ~train["engine"].isin(cal_set)
    mca =  train["engine"].isin(cal_set)
    sc  = StandardScaler().fit(train.loc[mtr, fc].values)
    return (sc.transform(train.loc[mtr, fc].values),
            train.loc[mtr, "RUL"].values,
            train.loc[mtr, "engine"].values,
            train.loc[mtr, "cycle"].values,
            sc.transform(train.loc[mca, fc].values),
            train.loc[mca, "RUL"].values,
            train.loc[mca, "engine"].values,
            train.loc[mca, "cycle"].values,
            sc.transform(test[fc].values),
            test["engine"].values,
            test["cycle"].values,
            np.clip(rul_t, 0, RUL_CLIP))

def make_seqs(X, y, engines, seq_len):
    seqs, labels = [], []
    for e in np.unique(engines):
        idx = np.where(engines == e)[0]
        Xe, ye = X[idx], y[idx]
        for end in range(seq_len, len(Xe)+1):
            seqs.append(Xe[end-seq_len:end])
            labels.append(ye[end-1])
    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)

def last_seqs(X, engines, cycles, seq_len, nf):
    out = []
    for e in np.unique(engines):
        idx = np.where(engines == e)[0]
        idx = idx[np.argsort(cycles[engines == e])]
        Xe  = X[idx]
        out.append(Xe[-seq_len:] if len(Xe) >= seq_len
                   else np.vstack([np.zeros((seq_len-len(Xe), nf)), Xe]))
    return np.array(out, dtype=np.float32)

def rand_cut_seqs(X, y, engines, cycles, seq_len, seed=7):
    rng = np.random.RandomState(seed)
    seqs, labels = [], []
    for e in np.unique(engines):
        idx = np.where(engines == e)[0]
        idx = idx[np.argsort(cycles[engines == e])]
        Xe, ye = X[idx], y[idx]
        n = len(Xe)
        if n < seq_len: continue
        end = rng.randint(seq_len, n)
        seqs.append(Xe[end-seq_len:end])
        labels.append(ye[end-1])
    return (np.array(seqs, dtype=np.float32),
            np.array(labels, dtype=np.float32))

# ── LSTM BASE MODEL ───────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.lstm = nn.LSTM(nf, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1, :]).squeeze(1)

def train_lstm(model, Xs, ys, epochs=30):
    loader = DataLoader(
        TensorDataset(torch.tensor(Xs), torch.tensor(ys).unsqueeze(1)),
        batch_size=256, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lf  = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = lf(model(xb), yb)
            loss.backward(); opt.step()
            total += loss.item()
        if (ep+1) % 10 == 0:
            print(f"    ep {ep+1}/{epochs} loss={total/len(loader):.2f}")
    return model

def predict_lstm(model, seqs):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), 512):
            xb = torch.tensor(seqs[i:i+512]).to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    return np.clip(np.concatenate(preds), 0, RUL_CLIP)

# ── CQR ───────────────────────────────────────────────────────────────────────

def run_cqr(train_preds, train_y, cal_preds, cal_y, test_preds, y_test,
            alpha=0.10):
    """
    Conformalized Quantile Regression (CQR) — Romano et al. 2019.

    CQR calibrates using:
      E_i = max(q_lo - y_i, y_i - q_hi)
    where q_lo, q_hi are alpha/2 and 1-alpha/2 quantile predictions.

    We implement CQR using the LSTM point predictions as a proxy:
    Since we have one model (not quantile regression), we use
    symmetric CQR: residuals = |y - pred|, same as standard CP
    but calibrate on TRAINING sequences (not last-cycle only).

    For true CQR we train quantile regressors at alpha/2 and 1-alpha/2.
    """
    from sklearn.linear_model import QuantileRegressor

    # Flatten sequences for quantile regression (use last frame features)
    # We use LSTM predictions as the single feature for quantile regression
    X_tr = train_preds.reshape(-1, 1)
    X_ca = cal_preds.reshape(-1, 1)
    X_te = test_preds.reshape(-1, 1)

    # Train lower and upper quantile regressors
    lo_alpha = alpha / 2      # 0.05
    hi_alpha = 1 - alpha / 2  # 0.95

    qr_lo = QuantileRegressor(quantile=lo_alpha, alpha=0.001, solver='highs')
    qr_hi = QuantileRegressor(quantile=hi_alpha, alpha=0.001, solver='highs')
    qr_lo.fit(X_tr, train_y)
    qr_hi.fit(X_tr, train_y)

    # Calibration conformity scores: E_i = max(q_lo(x) - y, y - q_hi(x))
    cal_lo = qr_lo.predict(X_ca)
    cal_hi = qr_hi.predict(X_ca)
    cal_scores = np.maximum(cal_lo - cal_y, cal_y - cal_hi)

    # Conformal quantile of scores
    n = len(cal_scores)
    q_cqr = float(np.quantile(cal_scores,
                               min(np.ceil((n+1)*(1-alpha))/n, 1.0)))

    # Test intervals
    te_lo = qr_lo.predict(X_te) - q_cqr
    te_hi = qr_hi.predict(X_te) + q_cqr

    cov  = ((y_test >= te_lo) & (y_test <= te_hi)).mean()
    wid  = (te_hi - te_lo).mean()

    # Late-life coverage (predicted RUL < 50)
    late = test_preds < 50
    if late.sum() > 0:
        lc = ((y_test[late] >= te_lo[late]) &
              (y_test[late] <= te_hi[late])).mean()
    else:
        lc = float('nan')

    return cov, wid, lc, q_cqr

# ── CONFORMAL HELPERS ─────────────────────────────────────────────────────────

def conf_q(resid, cov=0.90):
    n = len(resid)
    return float(np.quantile(resid, min(np.ceil((n+1)*cov)/n, 1.0)))

def cov_wid(y, lo, hi):
    return ((y>=lo)&(y<=hi)).mean(), (hi-lo).mean()

def late_cov(y, lo, hi, pred):
    late = pred < 50
    if late.sum() == 0: return float('nan')
    return ((y[late]>=lo[late])&(y[late]<=hi[late])).mean()

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("="*62)
    print("  CQR Baseline Experiment — Paper Tables 3 and 4")
    print("="*62)

    results = []

    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        print(f"\n{'='*55}")
        print(f"  {fd}")
        print(f"{'='*55}")

        (Xtr, ytr, etr, ctr,
         Xca, yca, eca, cca,
         Xte, ete, cte, y_test) = prepare(fd)
        nf = Xtr.shape[1]

        # Build sequences
        Xs, ys = make_seqs(Xtr, ytr, etr, SEQ_LEN)
        te_seqs = last_seqs(Xte, ete, cte, SEQ_LEN, nf)
        cal_last_seqs, cal_last_y = last_seqs(Xca, eca, cca, SEQ_LEN, nf), \
            np.array([yca[np.where(eca==e)[0][-1]] for e in np.unique(eca)])
        cal_rc_seqs, cal_rc_y = rand_cut_seqs(Xca, yca, eca, cca, SEQ_LEN)

        print(f"  Train seqs: {len(Xs):,}  Test: {len(te_seqs)}  "
              f"Cal (last): {len(cal_last_y)}  Cal (rc): {len(cal_rc_y)}")

        # Train LSTM
        print(f"  Training LSTM ...")
        lstm = LSTMModel(nf).to(DEVICE)
        lstm = train_lstm(lstm, Xs, ys, epochs=30)

        # Predict
        print(f"  Predicting ...")
        train_preds  = predict_lstm(lstm, Xs)
        te_preds     = predict_lstm(lstm, te_seqs)
        cal_last_pred= predict_lstm(lstm, cal_last_seqs)
        cal_rc_pred  = predict_lstm(lstm, cal_rc_seqs)

        # 1. Standard CP (last-cycle calibration)
        resid_std = np.abs(cal_last_y - cal_last_pred)
        q_std     = conf_q(resid_std)
        lo_std, hi_std = te_preds - q_std, te_preds + q_std
        cov_std, wid_std = cov_wid(y_test, lo_std, hi_std)
        lc_std = late_cov(y_test, lo_std, hi_std, te_preds)

        # 2. CQR
        cov_cqr, wid_cqr, lc_cqr, q_cqr = run_cqr(
            train_preds, ys, cal_rc_pred, cal_rc_y,
            te_preds, y_test)

        # 3. Global random-cutoff CP
        resid_rc = np.abs(cal_rc_y - cal_rc_pred)
        q_rc     = conf_q(resid_rc)
        lo_rc, hi_rc = te_preds - q_rc, te_preds + q_rc
        cov_rc, wid_rc = cov_wid(y_test, lo_rc, hi_rc)
        lc_rc = late_cov(y_test, lo_rc, hi_rc, te_preds)

        mae = mean_absolute_error(y_test, te_preds)

        print(f"\n  MAE = {mae:.2f}")
        print(f"\n  {'Method':<22}{'Coverage':>10}{'Width':>8}"
              f"{'Late-cov':>10}{'q':>8}  Honest?")
        print(f"  {'-'*60}")
        for name, cov, wid, lc, q in [
            ('Standard CP',      cov_std, wid_std, lc_std, q_std),
            ('CQR',              cov_cqr, wid_cqr, lc_cqr, q_cqr),
            ('Random-cutoff CP', cov_rc,  wid_rc,  lc_rc,  q_rc),
        ]:
            h = 'YES' if abs(cov-COVERAGE)<0.10 else 'NO'
            lc_str = f'{lc:.1%}' if not np.isnan(lc) else 'nan'
            print(f"  {name:<22}{cov:>9.1%}{wid:>8.1f}"
                  f"{lc_str:>10}{q:>8.2f}  {h}")

        results.append(dict(
            fd=fd, mae=round(mae,2),
            std_cov=f"{cov_std:.0%}", std_wid=round(wid_std,1),
            std_q=round(q_std,1),    std_lc=f"{lc_std:.0%}",
            cqr_cov=f"{cov_cqr:.0%}",cqr_wid=round(wid_cqr,1),
            cqr_q=round(q_cqr,1),   cqr_lc=f"{lc_cqr:.0%}" if not np.isnan(lc_cqr) else 'nan',
            rc_cov=f"{cov_rc:.0%}",  rc_wid=round(wid_rc,1),
            rc_q=round(q_rc,1),      rc_lc=f"{lc_rc:.0%}",
        ))

    # Final table for paper
    print(f"\n\n{'='*75}")
    print("  COPY THESE VALUES INTO PAPER TABLES 3 AND 4")
    print(f"{'='*75}")
    print(f"\n  TABLE 3 — Coverage comparison:")
    print(f"  {'FD':<6}{'Std cov':>9}{'Std wid':>9}{'Std q':>7}"
          f"{'CQR cov':>9}{'CQR wid':>9}{'CQR q':>7}{'RC cov':>9}")
    print(f"  {'-'*65}")
    for r in results:
        print(f"  {r['fd']:<6}{r['std_cov']:>9}{r['std_wid']:>9}"
              f"{r['std_q']:>7}{r['cqr_cov']:>9}{r['cqr_wid']:>9}"
              f"{r['cqr_q']:>7}{r['rc_cov']:>9}")

    print(f"\n  TABLE 4 — Late-life coverage (RUL < 50):")
    print(f"  {'FD':<6}{'Std late':>10}{'CQR late':>10}{'RC late':>10}")
    print(f"  {'-'*38}")
    for r in results:
        print(f"  {r['fd']:<6}{r['std_lc']:>10}{r['cqr_lc']:>10}{r['rc_lc']:>10}")
    print(f"{'='*75}")
    print("\n  Replace the [NOTE] placeholder values in Paper_Final.docx")
    print("  with these real numbers before sending to professor.")

if __name__ == "__main__":
    main()
