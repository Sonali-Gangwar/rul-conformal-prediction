"""
proposal2_clean.py — Proposal 2: Domain-Specific CNN+GRU vs Chronos-2

CLEAN RESEARCH QUESTION:
  Does a small architecture (235K params) specifically designed for
  degradation signals beat a large general foundation model (Chronos-2,
  710M params) on RUL prediction?

  This tests: architectural inductive bias vs scale and pre-training breadth.

  CNN captures local sensor patterns (short-range degradation trends).
  GRU captures temporal evolution (how degradation progresses over time).
  Both are well-suited to degradation signals — unlike Chronos-2 which
  was designed for general time series.

COMPARISON:
  Proposal 1: Chronos-2 (frozen, 710M) + NN head
  Proposal 2: CNN+GRU (trained from scratch, 235K) + conformal calibration
  Baseline:   LSTM ensemble (from published paper)

Run: python proposal2_clean.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN  = 30
RUL_CLIP = 125
COVERAGE = 0.90
DATA_DIR = r"D:\rul_project\data"
COLS     = ["engine","cycle","op1","op2","op3"] + [f"s{i}" for i in range(1,22)]
DROP     = {"s1","s5","s6","s10","s16","s18","s19"}
print(f"Device: {DEVICE}")

# ── DATA ─────────────────────────────────────────────────────────────────────

def load_cmapss(fd, split="train"):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{split}_{fd}.txt"),
                     sep=r"\s+", header=None).iloc[:, :26]
    df.columns = COLS
    fc = [c for c in COLS if c.startswith("s") and c not in DROP]
    if split == "train":
        mc = df.groupby("engine")["cycle"].transform("max")
        df["RUL"] = (mc - df["cycle"]).clip(upper=RUL_CLIP)
    sc = StandardScaler()
    X  = sc.fit_transform(df[fc].values).astype(np.float32)
    rul = df["RUL"].values if "RUL" in df.columns else None
    return X, df["engine"].values, df["cycle"].values, rul

def make_rul_windows(X, y, engines, seq_len):
    seqs, labels = [], []
    for e in np.unique(engines):
        idx = np.where(engines == e)[0]
        Xe, ye = X[idx], y[idx]
        for end in range(seq_len, len(Xe)+1):
            seqs.append(Xe[end-seq_len:end])
            labels.append(ye[end-1])
    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)

def last_windows(X, engines, cycles, seq_len, nf):
    out = []
    for e in np.unique(engines):
        idx = np.where(engines == e)[0]
        idx = idx[np.argsort(cycles[engines == e])]
        Xe  = X[idx]
        out.append(Xe[-seq_len:] if len(Xe) >= seq_len
                   else np.vstack([np.zeros((seq_len-len(Xe), nf)), Xe]))
    return np.array(out, dtype=np.float32)

def multi_rand_cut(X, y, engines, cycles, seq_len, n_cuts=5):
    rng_list = [np.random.RandomState(s) for s in range(n_cuts)]
    all_seqs, all_y = [], []
    for e in np.unique(engines):
        idx = np.where(engines == e)[0]
        idx = idx[np.argsort(cycles[engines == e])]
        Xe, ye = X[idx], y[idx]
        n = len(Xe)
        if n < seq_len: continue
        for rng in rng_list:
            end = rng.randint(seq_len, n)
            all_seqs.append(Xe[end-seq_len:end])
            all_y.append(ye[end-1])
    return np.array(all_seqs, dtype=np.float32), np.array(all_y, dtype=np.float32)

# ── CNN+GRU MODEL ─────────────────────────────────────────────────────────────

class CNNGRU(nn.Module):
    """
    CNN+GRU model trained FROM SCRATCH on C-MAPSS.
    CNN: captures local sensor patterns (3-cycle windows)
    GRU: captures temporal degradation evolution
    This architecture is specifically designed for degradation signals.
    """
    def __init__(self, n_features, d_model=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.gru = nn.GRU(128, d_model, num_layers=2,
                          batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, seq_len, n_features)
        x_t = x.transpose(1, 2)           # (B, n_features, seq_len)
        c   = self.cnn(x_t).transpose(1, 2)  # (B, seq_len, 128)
        _, h = self.gru(c)                 # h: (2, B, d_model)
        return self.head(h[-1]).squeeze(1) # (B,)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ── TRAINING ─────────────────────────────────────────────────────────────────

def train_model(model, Xs, ys, epochs=60, batch=256, lr=1e-3):
    loader = DataLoader(
        TensorDataset(torch.tensor(Xs, dtype=torch.float32),
                      torch.tensor(ys, dtype=torch.float32)),
        batch_size=batch, shuffle=True)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    lf    = nn.MSELoss()
    model.train()

    for ep in range(epochs):
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = lf(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()
        if (ep+1) % 15 == 0:
            print(f"    ep {ep+1}/{epochs}  loss={total/len(loader):.2f}")

    return model

def predict(model, X, batch=256):
    model.eval()
    preds = []
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(Xt), batch):
            preds.append(model(Xt[i:i+batch].to(DEVICE)).cpu().numpy())
    return np.clip(np.concatenate(preds), 0, RUL_CLIP)

# ── CONFORMAL ────────────────────────────────────────────────────────────────

def conf_q(r, cov=0.90):
    n = len(r)
    return float(np.quantile(r, min(np.ceil((n+1)*cov)/n, 1.0)))

def cov_wid(y, lo, hi):
    return ((y>=lo)&(y<=hi)).mean(), (hi-lo).mean()

def late_cov(y, lo, hi, pred):
    late = pred < 50
    if late.sum() == 0: return float('nan')
    return ((y[late]>=lo[late])&(y[late]<=hi[late])).mean()

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("="*62)
    print("  PROPOSAL 2: CNN+GRU (scratch) vs Chronos-2 (Proposal 1)")
    print("="*62)
    print(f"\n  Research question: Does domain-specific architecture")
    print(f"  (235K params) beat general foundation model (710M params)?")

    results = []

    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        print(f"\n{'='*62}")
        print(f"  {fd}")
        print(f"{'='*62}")

        X, engines, cycles, rul = load_cmapss(fd, "train")
        Xt, et, ct, _           = load_cmapss(fd, "test")
        y_test = np.clip(
            pd.read_csv(os.path.join(DATA_DIR, f"RUL_{fd}.txt"),
                        header=None).iloc[:,0].values, 0, RUL_CLIP)

        # Split
        all_e = np.unique(engines)
        np.random.seed(42); np.random.shuffle(all_e)
        n_cal   = max(5, int(len(all_e)*0.20))
        cal_set = set(all_e[:n_cal])
        mtr = ~np.isin(engines, list(cal_set))
        mca =  np.isin(engines, list(cal_set))

        ytr = rul[mtr]; yca = rul[mca]
        eca = engines[mca]; cca = cycles[mca]
        nf  = X.shape[1]

        # Build sequences
        Xs, ys = make_rul_windows(X[mtr], ytr, engines[mtr], SEQ_LEN)
        print(f"  Train seqs: {len(Xs):,}  features: {nf}")

        # Train CNN+GRU from scratch
        print(f"  Training CNN+GRU from scratch ({60} epochs) ...")
        model = CNNGRU(nf).to(DEVICE)
        print(f"  Parameters: {model.count_params():,}")
        model = train_model(model, Xs, ys, epochs=60)

        # Predict
        te_seqs  = last_windows(Xt, et, ct, SEQ_LEN, nf)
        cal_seqs, cal_y = multi_rand_cut(X[mca], yca, eca, cca, SEQ_LEN)

        pred     = predict(model, te_seqs)
        cal_pred = predict(model, cal_seqs)

        # Check for NaN
        if np.isnan(pred).any():
            print(f"  ERROR: NaN in predictions — skipping {fd}")
            continue

        mae  = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2   = r2_score(y_test, pred)

        resid    = np.abs(cal_y - cal_pred)
        q        = conf_q(resid)
        lo, hi   = pred - q, pred + q
        cov, wid = cov_wid(y_test, lo, hi)
        lc       = late_cov(y_test, lo, hi, pred)
        h        = "YES" if abs(cov-COVERAGE) < 0.10 else "NO"

        print(f"\n  MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}")
        print(f"  Cov={cov:.0%}  Wid={wid:.1f}  Late={lc:.0%}  Honest={h}")

        results.append(dict(fd=fd, MAE=round(mae,2), RMSE=round(rmse,2),
                            R2=round(r2,3), cov=f"{cov:.0%}",
                            wid=round(wid,1), late=f"{lc:.0%}", honest=h))

    # Final comparison
    print(f"\n\n{'='*72}")
    print("  FINAL — CNN+GRU (P2) vs Chronos-2 (P1) vs LSTM baseline")
    print(f"{'='*72}")
    print(f"  {'FD':<6}{'P2 CNN+GRU':>13}{'P1 Chronos-2':>14}"
          f"{'LSTM':>10}{'P2 Cov':>10}  Better than P1?")
    print(f"  {'-'*60}")
    p1   = {'FD001':'15.17','FD002':'19.16','FD003':'12.71','FD004':'19.03'}
    lstm = {'FD001':'10.42','FD002':'11.95','FD003':'11.88','FD004':'13.45'}
    for r in results:
        b = float(r['MAE']) < float(p1[r['fd']])
        print(f"  {r['fd']:<6}{r['MAE']:>13}{p1[r['fd']]:>14}"
              f"{lstm[r['fd']]:>10}{r['cov']:>10}  "
              f"{'YES <--' if b else 'no'}")
    print(f"{'='*72}")
    print()
    print("  Params: CNN+GRU=235K  vs  Chronos-2=710M")
    print("  KEY: Can a small domain-specific model beat a")
    print("  large general foundation model on degradation data?")
    print(f"{'='*72}")

if __name__ == "__main__":
    main()
