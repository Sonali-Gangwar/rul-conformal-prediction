"""
chronos_rul_v4.py — Proposal 1 WORKING: Chronos-2 + Conformal RUL

CONFIRMED API from check_chronos.py:
  tok = tokenizer.context_input_transform(batch)  # batch shape (B, seq_len)
  tok[0] shape: (B, seq_len_tok)  e.g. (3, 31)
  tok[1] shape: (B, seq_len_tok)

  enc = model.encode(tok[0], tok[1])
  enc is a TENSOR shape (B, seq_len_tok, 256) -- NOT a tuple
  enc[i] gives the i-th SAMPLE embedding of shape (seq_len_tok, 256)

  So for batch of B windows:
    enc shape: (B, seq_len_tok, d_model)
    mean over seq_len_tok -> (B, d_model)
    collect each of B rows -> B items of shape (d_model,)

Run: python chronos_rul_v4.py
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from chronos import BaseChronosPipeline

DATA_DIR = "data"
SEQ_LEN  = 30
RUL_CLIP = 125
COVERAGE = 0.90
COLS     = ["engine","cycle","op1","op2","op3"] + [f"s{i}" for i in range(1,22)]
DROP     = {"s1","s5","s6","s10","s16","s18","s19"}

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

def extract_embeddings(pipeline, windows, batch_size=32):
    """
    CONFIRMED working API for Chronos v2.3.1:

    enc = model.encode(input_ids, attention_mask)
    enc is a TENSOR of shape (B, seq_len_tok, 256)
    NOT a tuple.

    Steps:
    1. enc.mean(dim=1) -> (B, 256)  average over token dimension
    2. .cpu().numpy() -> (B, 256) numpy
    3. collect each row i individually -> list of B items each (256,)
    4. np.array(list) -> (N_total, 256)
    5. hstack across sensors -> (N_total, 256*n_sensors)
    """
    N, sl, nf = windows.shape
    n_sensors  = min(nf, 5)
    all_sensor_embs = []

    for feat_i in range(n_sensors):
        series_all = torch.tensor(
            windows[:, :, feat_i], dtype=torch.float32)  # (N, seq_len)
        rows = []

        for start in range(0, N, batch_size):
            batch = series_all[start:start + batch_size]  # (B, seq_len)
            B = batch.shape[0]

            tok            = pipeline.tokenizer.context_input_transform(batch)
            input_ids      = tok[0]   # (B, seq_len_tok)
            attention_mask = tok[1]   # (B, seq_len_tok)

            with torch.no_grad():
                enc = pipeline.model.encode(input_ids, attention_mask)
                # enc is a TENSOR (B, seq_len_tok, 256)
                # mean over seq_len_tok -> (B, 256)
                emb = enc.mean(dim=1)   # (B, 256)

            emb_np = emb.cpu().numpy()  # (B, 256) numpy array

            # collect each row individually
            for i in range(B):
                rows.append(emb_np[i])  # shape (256,)

        sensor_matrix = np.array(rows)   # (N, 256)
        all_sensor_embs.append(sensor_matrix)

    return np.hstack(all_sensor_embs)    # (N, 256*n_sensors)

def conf_q(resid, cov=0.90):
    n = len(resid)
    return float(np.quantile(resid, min(np.ceil((n+1)*cov)/n, 1.0)))

def cov_wid(y, lo, hi):
    return ((y>=lo)&(y<=hi)).mean(), (hi-lo).mean()

def late_cov(y, lo, hi, pred):
    late = pred < 50
    if late.sum() == 0: return float('nan')
    return ((y[late]>=lo[late])&(y[late]<=hi[late])).mean()

def run_fd(fd, pipeline):
    print(f"\n{'='*60}")
    print(f"  {fd} — Chronos-2 + Conformal")
    print(f"{'='*60}")

    (Xtr,ytr,etr,ctr, Xca,yca,eca,cca,
     Xte,ete,cte,y_test) = prepare(fd)
    nf = Xtr.shape[1]

    Xs, ys       = make_windows(Xtr, ytr, etr, SEQ_LEN)
    te_wins      = last_windows(Xte, ete, cte, SEQ_LEN, nf)
    cal_wins, cy = random_cut_windows(Xca,yca,eca,cca,SEQ_LEN,nf)

    print(f"  train={len(Xs):,}  test={len(te_wins)}  cal={len(cal_wins)}")

    print("  Extracting train embeddings ...")
    tr_emb = extract_embeddings(pipeline, Xs)
    ok = tr_emb.shape[0] == len(ys)
    print(f"  shape={tr_emb.shape}  {'OK ✓' if ok else 'MISMATCH ✗'}")
    if not ok:
        raise ValueError(f"Train mismatch: {tr_emb.shape[0]} vs {len(ys)}")

    print("  Extracting test embeddings ...")
    te_emb = extract_embeddings(pipeline, te_wins)

    print("  Extracting calibration embeddings ...")
    ca_emb = extract_embeddings(pipeline, cal_wins)

    esc      = StandardScaler().fit(tr_emb)
    ridge    = Ridge(alpha=1.0).fit(esc.transform(tr_emb), ys)
    pred     = np.clip(ridge.predict(esc.transform(te_emb)), 0, RUL_CLIP)
    cal_pred = np.clip(ridge.predict(esc.transform(ca_emb)), 0, RUL_CLIP)

    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)

    resid    = np.abs(cy - cal_pred)
    q        = conf_q(resid)
    lo, hi   = pred-q, pred+q
    cov, wid = cov_wid(y_test, lo, hi)
    lc       = late_cov(y_test, lo, hi, pred)
    h        = "YES" if abs(cov-COVERAGE)<0.10 else "NO"

    print(f"\n  MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f}")
    print(f"  Cov={cov:.0%} Wid={wid:.1f} Late={lc:.0%} q={q:.1f} Honest={h}")

    return dict(fd=fd,MAE=round(mae,2),RMSE=round(rmse,2),R2=round(r2,3),
                cov=f"{cov:.0%}",wid=round(wid,1),
                late=f"{lc:.0%}",q=round(q,1),honest=h)


def main():
    print("="*60)
    print("  Chronos-2 Foundation Model + Conformal RUL (v4)")
    print("="*60)
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cpu",
        torch_dtype=torch.float32)
    print("  Loaded.\n")

    results = []
    for fd in ["FD001","FD002","FD003","FD004"]:
        try:
            results.append(run_fd(fd, pipeline))
        except Exception as ex:
            print(f"  ERROR {fd}: {ex}")
            import traceback; traceback.print_exc()

    print(f"\n\n{'='*70}")
    print("  FINAL — Chronos-2 Foundation Model + Conformal")
    print(f"{'='*70}")
    print(f"  {'FD':<6}{'MAE':>7}{'RMSE':>7}{'R2':>7}"
          f"{'Cov':>8}{'Wid':>7}{'Late':>8}  Honest?")
    print(f"  {'-'*58}")
    for r in results:
        print(f"  {r['fd']:<6}{r['MAE']:>7}{r['RMSE']:>7}{r['R2']:>7}"
              f"{r['cov']:>8}{r['wid']:>7}{r['late']:>8}  {r['honest']}")
    print(f"{'='*70}")
    print("\n  LSTM baseline:")
    print("  FD001=10.42 | FD002=11.95 | FD003=11.88 | FD004=13.45")
    print("\n  ONE frozen Chronos-2 — no retraining between subsets.")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
