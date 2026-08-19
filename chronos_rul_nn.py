"""
chronos_rul_nn.py — Proposal 1 Step 1: Chronos-2 + Neural Network Head + Conformal

IMPROVEMENT OVER v4:
  v4 used Ridge regression as the RUL head -> MAE 13-20 cycles
  This version uses a small 2-layer neural network head -> expected MAE 10-14 cycles

CONFIRMED API (from chronos_rul_v4.py):
  enc = model.encode(input_ids, attention_mask)  -> tensor (B, seq_len_tok, 256)
  enc.mean(dim=1) -> (B, 256)
  collect rows one by one -> (N, 256) per sensor
  hstack 5 sensors -> (N, 1280)

Run: python chronos_rul_nn.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from chronos import BaseChronosPipeline

DATA_DIR = "data"
SEQ_LEN  = 30
RUL_CLIP = 125
COVERAGE = 0.90
COLS     = ["engine","cycle","op1","op2","op3"] + [f"s{i}" for i in range(1,22)]
DROP     = {"s1","s5","s6","s10","s16","s18","s19"}

# ── DATA ─────────────────────────────────────────────────────────────────────

def load_fd(split, fd):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{split}_{fd}.txt"),
                     sep=r"\s+", header=None).iloc[:, :26]
    df.columns = COLS
    return df

def prepare(fd):
    train = load_fd("train", fd)
    test  = load_fd("test",  fd)
    rul   = pd.read_csv(os.path.join(DATA_DIR, f"RUL_{fd}.txt"),
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
            np.clip(rul, 0, RUL_CLIP))

# ── SEQUENCES ────────────────────────────────────────────────────────────────

def make_windows(X, y, engines, seq_len):
    seqs, labels = [], []
    for e in np.unique(engines):
        idx = np.where(engines == e)[0]
        Xe, ye = X[idx], y[idx]
        for end in range(seq_len, len(Xe) + 1):
            seqs.append(Xe[end - seq_len:end])
            labels.append(ye[end - 1])
    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)

def last_windows(X, engines, cycles, seq_len, nf):
    out = []
    for e in np.unique(engines):
        idx = np.where(engines == e)[0]
        idx = idx[np.argsort(cycles[engines == e])]
        Xe  = X[idx]
        if len(Xe) >= seq_len:
            out.append(Xe[-seq_len:])
        else:
            out.append(np.vstack([np.zeros((seq_len-len(Xe), nf),
                                           dtype=np.float32), Xe]))
    return np.array(out, dtype=np.float32)

def random_cut_windows(X, y, engines, cycles, seq_len, nf, seed=7):
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
        seqs.append(Xe[end - seq_len:end])
        labels.append(ye[end - 1])
    return (np.array(seqs, dtype=np.float32),
            np.array(labels, dtype=np.float32))

# ── CHRONOS EMBEDDINGS (confirmed working API from v4) ────────────────────────

def extract_embeddings(pipeline, windows, batch_size=32):
    N, sl, nf = windows.shape
    n_sensors  = min(nf, 5)
    all_sensor_embs = []

    for feat_i in range(n_sensors):
        series_all = torch.tensor(
            windows[:, :, feat_i], dtype=torch.float32)
        rows = []

        for start in range(0, N, batch_size):
            batch = series_all[start:start + batch_size]
            B     = batch.shape[0]
            tok            = pipeline.tokenizer.context_input_transform(batch)
            input_ids      = tok[0]
            attention_mask = tok[1]
            with torch.no_grad():
                enc = pipeline.model.encode(input_ids, attention_mask)
                emb = enc.mean(dim=1)   # (B, d_model)
            emb_np = emb.cpu().numpy()
            for i in range(B):
                rows.append(emb_np[i])

        sensor_matrix = np.array(rows)
        all_sensor_embs.append(sensor_matrix)

    return np.hstack(all_sensor_embs)   # (N, d_model*n_sensors)

# ── NEURAL NETWORK RUL HEAD ───────────────────────────────────────────────────

class RULHead(nn.Module):
    """
    Small 2-layer MLP on top of frozen Chronos-2 embeddings.
    Input: (batch, embed_dim)  e.g. (batch, 1280)
    Output: (batch, 1)  predicted RUL
    """
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_head(emb_train, y_train, in_dim, device, epochs=50, batch_size=512):
    """Train the neural network RUL head on Chronos embeddings."""
    model = RULHead(in_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    lf    = nn.MSELoss()

    X_t = torch.tensor(emb_train, dtype=torch.float32)
    y_t = torch.tensor(y_train,   dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=batch_size, shuffle=True)

    model.train()
    for ep in range(epochs):
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lf(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1}/{epochs}  loss={total/len(loader):.2f}")

    return model


def predict_head(model, emb, device, batch_size=512):
    model.eval()
    preds = []
    X_t = torch.tensor(emb, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i+batch_size].to(device)
            preds.append(model(xb).cpu().numpy())
    return np.clip(np.concatenate(preds), 0, RUL_CLIP)

# ── CONFORMAL ────────────────────────────────────────────────────────────────

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

def run_fd(fd, pipeline, device):
    print(f"\n{'='*62}")
    print(f"  {fd} — Chronos-2 + NN Head + Conformal")
    print(f"{'='*62}")

    (Xtr,ytr,etr,ctr, Xca,yca,eca,cca,
     Xte,ete,cte,y_test) = prepare(fd)
    nf = Xtr.shape[1]

    Xs, ys       = make_windows(Xtr, ytr, etr, SEQ_LEN)
    te_wins      = last_windows(Xte, ete, cte, SEQ_LEN, nf)
    cal_wins, cy = random_cut_windows(Xca, yca, eca, cca, SEQ_LEN, nf)

    print(f"  train={len(Xs):,}  test={len(te_wins)}  cal={len(cal_wins)}")

    # Extract Chronos-2 embeddings (frozen)
    print("  Extracting train embeddings (Chronos-2 frozen) ...")
    tr_emb = extract_embeddings(pipeline, Xs)
    print(f"  shape={tr_emb.shape}  OK")

    print("  Extracting test embeddings ...")
    te_emb = extract_embeddings(pipeline, te_wins)

    print("  Extracting calibration embeddings ...")
    ca_emb = extract_embeddings(pipeline, cal_wins)

    # Normalise embeddings
    esc    = StandardScaler().fit(tr_emb)
    tr_s   = esc.transform(tr_emb)
    te_s   = esc.transform(te_emb)
    ca_s   = esc.transform(ca_emb)

    # Train neural network head
    in_dim = tr_s.shape[1]
    print(f"  Training NN head (in_dim={in_dim}, 50 epochs) ...")
    head   = train_head(tr_s, ys, in_dim, device, epochs=50)

    # Predict
    pred     = predict_head(head, te_s, device)
    cal_pred = predict_head(head, ca_s, device)

    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)

    # Conformal calibration
    resid    = np.abs(cy - cal_pred)
    q        = conf_q(resid)
    lo, hi   = pred - q, pred + q
    cov, wid = cov_wid(y_test, lo, hi)
    lc       = late_cov(y_test, lo, hi, pred)
    h        = "YES" if abs(cov - COVERAGE) < 0.10 else "NO"

    print(f"\n  MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}")
    print(f"  Cov={cov:.0%}  Wid={wid:.1f}  Late={lc:.0%}  q={q:.1f}  Honest={h}")

    return dict(fd=fd, MAE=round(mae,2), RMSE=round(rmse,2), R2=round(r2,3),
                cov=f"{cov:.0%}", wid=round(wid,1),
                late=f"{lc:.0%}", q=round(q,1), honest=h)


def main():
    print("="*62)
    print("  Proposal 1: Chronos-2 + NN Head + Conformal RUL")
    print("="*62)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    print("  Loading Chronos-2 ...")
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cpu",
        torch_dtype=torch.float32)
    print("  Loaded.\n")

    results = []
    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        try:
            results.append(run_fd(fd, pipeline, device))
        except Exception as ex:
            print(f"  ERROR {fd}: {ex}")
            import traceback; traceback.print_exc()

    print(f"\n\n{'='*72}")
    print("  FINAL — Chronos-2 + NN Head + Conformal (Proposal 1)")
    print(f"{'='*72}")
    print(f"  {'FD':<6}{'MAE':>7}{'RMSE':>7}{'R2':>7}"
          f"{'Cov':>8}{'Wid':>7}{'Late':>8}  Honest?")
    print(f"  {'-'*62}")
    for r in results:
        print(f"  {r['fd']:<6}{r['MAE']:>7}{r['RMSE']:>7}{r['R2']:>7}"
              f"{r['cov']:>8}{r['wid']:>7}{r['late']:>8}  {r['honest']}")
    print(f"{'='*72}")
    print()
    print("  Ridge baseline (v4):")
    print("  FD001=17.12 | FD002=20.48 | FD003=13.79 | FD004=19.74")
    print()
    print("  LSTM baseline (your existing paper):")
    print("  FD001=10.42 | FD002=11.95 | FD003=11.88 | FD004=13.45")
    print()
    print("  Target: Chronos-2+NN MAE should approach or beat LSTM.")
    print("  If yes -> foundation model is competitive with trained LSTM.")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
