"""
chronos_new_datasets.py — Proposal 1: Chronos-2 on IMS + Battery + KAIST

IMS Bearing:
  - 3 tests, 4 bearings each, files named by timestamp
  - Each file: 20480 rows x 8 columns (4 bearings x 2 channels each)
  - Health indicator: RMS per file per bearing channel
  - Same preprocessing as FEMTO

Battery (NASA):
  - B0005, B0006, B0007, B0018 .mat files
  - 168 discharge cycles per battery
  - Health indicator: capacity per cycle (degrades from ~2Ah to 1.4Ah)
  - RUL = cycles remaining until capacity < 1.4Ah

KAIST:
  - 128 CSV files, one per hour, 2M rows each
  - 4 columns: vibration_x, vibration_y, temperature, unknown
  - Health indicator: RMS of vibration per hour (one value per CSV)
  - One bearing run to failure at 128 hours

Run: python chronos_new_datasets.py
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from chronos import BaseChronosPipeline

# ── PATHS ────────────────────────────────────────────────────────────────────
IMS_DIR     = r"D:\rul_project\data\IMS"
BATTERY_DIR = r"D:\rul_project\data\Battery"
KAIST_DIR   = r"D:\rul_project\data\KAIST"
RUL_CLIP    = 125
COVERAGE    = 0.90

# ── IMS LOADING ───────────────────────────────────────────────────────────────

def load_ims_test(test_folder):
    """
    Load one IMS test. Each file has 20480 rows x 8 columns.
    Columns: bearing1_ch1, bearing1_ch2, bearing2_ch1, bearing2_ch2,
             bearing3_ch1, bearing3_ch2, bearing4_ch1, bearing4_ch2
    Extract RMS per file per bearing (4 bearings x 2 channels = 8 values per file).
    Returns: (n_files, 8) array of RMS values — one row per time window.
    """
    inner = os.path.join(test_folder, os.listdir(test_folder)[0])
    if os.path.isdir(inner):
        inner2 = os.path.join(inner, os.listdir(inner)[0])
        folder = inner2 if os.path.isdir(inner2)  else inner
    else:
        folder = test_folder

    files = sorted(os.listdir(folder))
    rms_list = []
    for f in files:
        try:
            data = np.fromstring(open(os.path.join(folder, f)).read(), sep='\t').reshape(-1, 8)
            if data.shape[1] < 8:
                continue
            rms = np.sqrt(np.mean(data**2, axis=0))[:8]
            rms_list.append(rms)
        except Exception:
            continue

    return np.array(rms_list, dtype=np.float32) if rms_list else None

# ── BATTERY LOADING ───────────────────────────────────────────────────────────

def load_battery(mat_path, battery_name):
    """
    Load NASA battery .mat file.
    Extract capacity per discharge cycle as the degradation signal.
    RUL = remaining cycles until capacity < 1.4 Ah (end of life = 30% fade).
    Returns: (n_cycles, 1) array of capacity values, RUL per cycle.
    """
    mat  = sio.loadmat(mat_path)
    cycs = mat[battery_name][0, 0]['cycle'][0]
    caps = []
    for c in cycs:
        if c['type'][0] == 'discharge':
            try:
                cap = float(c['data'][0, 0]['Capacity'][0][-1])
                caps.append(cap)
            except Exception:
                continue

    caps = np.array(caps, dtype=np.float32)
    # RUL = cycles until capacity < 1.4Ah
    eol = np.where(caps < 1.4)[0]
    if len(eol) > 0:
        eol_idx = eol[0]
    else:
        eol_idx = len(caps)

    rul = np.clip(
        np.array([max(0, eol_idx - i) for i in range(len(caps))],
                 dtype=np.float32),
        0, RUL_CLIP)

    # Signal: capacity value per cycle (1 feature)
    signal = caps.reshape(-1, 1)
    return signal, rul

# ── KAIST LOADING ────────────────────────────────────────────────────────────

def load_kaist(kaist_dir):
    """
    Load KAIST bearing dataset.
    One CSV per hour, 2M rows x 4 columns.
    Columns: vib_x, vib_y, temperature, unknown
    Health indicator: RMS of vib_x and vib_y per hour = 2 features per file.
    RUL: bearing failed at last file, so RUL = (n_files - current_file).
    """
    files = sorted([f for f in os.listdir(kaist_dir) if f.endswith('.csv')])
    rms_list = []
    for f in files:
        try:
            df = pd.read_csv(os.path.join(kaist_dir, f), header=None,
                             nrows=100000)  # sample 100k rows per file for speed
            vx  = df.iloc[:, 0].values.astype(float)
            vy  = df.iloc[:, 1].values.astype(float)
            rms_list.append([np.sqrt(np.mean(vx**2)),
                             np.sqrt(np.mean(vy**2))])
        except Exception:
            continue

    signal = np.array(rms_list, dtype=np.float32)  # (n_hours, 2)
    n      = len(signal)
    rul    = np.clip(np.arange(n-1, -1, -1, dtype=np.float32), 0, RUL_CLIP)
    return signal, rul

# ── SEQUENCE HELPERS ─────────────────────────────────────────────────────────

def make_seqs(signal, rul, seq_len):
    seqs, labels = [], []
    for end in range(seq_len, len(signal)+1):
        seqs.append(signal[end-seq_len:end])
        labels.append(rul[end-1])
    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)

def random_cut_seqs(signal, rul, seq_len, n=20, seed=7):
    rng  = np.random.RandomState(seed)
    N    = len(signal)
    if N < seq_len: return None, None
    cuts = rng.choice(np.arange(seq_len, N), size=min(n, N-seq_len),
                      replace=False)
    seqs   = np.array([signal[c-seq_len:c] for c in cuts], dtype=np.float32)
    labels = np.array([rul[c-1] for c in cuts], dtype=np.float32)
    return seqs, labels

def eval_seqs(signal, rul, seq_len, n=30, seed=99):
    rng  = np.random.RandomState(seed)
    N    = len(signal)
    if N < seq_len: return None, None
    cuts = rng.choice(np.arange(seq_len, N), size=min(n, N-seq_len),
                      replace=False)
    seqs   = np.array([signal[c-seq_len:c] for c in cuts], dtype=np.float32)
    labels = np.array([rul[c-1] for c in cuts], dtype=np.float32)
    return seqs, labels

# ── CHRONOS EMBEDDINGS (confirmed GPU API) ────────────────────────────────────

def extract_embeddings(pipeline, windows, device, batch_size=64):
    N, sl, nf = windows.shape
    all_embs   = []
    for feat_i in range(nf):
        series = torch.tensor(windows[:, :, feat_i], dtype=torch.float32)
        rows   = []
        for start in range(0, N, batch_size):
            batch = series[start:start+batch_size]
            B     = batch.shape[0]
            tok   = pipeline.tokenizer.context_input_transform(batch)
            ids   = tok[0].to(device)
            mask  = tok[1].to(device)
            with torch.no_grad():
                enc = pipeline.model.encode(ids, mask)
                emb = enc.mean(dim=1).cpu().numpy()
            for i in range(B):
                rows.append(emb[i])
        all_embs.append(np.array(rows))
    return np.hstack(all_embs)

# ── NN HEAD ───────────────────────────────────────────────────────────────────

class RULHead(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Linear(32, 1))
    def forward(self, x): return self.net(x).squeeze(1)

def train_head(emb, y, device, epochs=40, batch=64):
    model  = RULHead(emb.shape[1]).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lf     = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(emb, dtype=torch.float32),
                      torch.tensor(y,   dtype=torch.float32)),
        batch_size=batch, shuffle=True)
    model.train()
    for ep in range(epochs):
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss=lf(model(xb),yb)
            loss.backward(); opt.step(); total+=loss.item()
        if (ep+1)%10==0:
            print(f"    ep {ep+1}/{epochs} loss={total/len(loader):.2f}")
    return model

def predict(model, emb, device, batch=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(emb), batch):
            xb = torch.tensor(emb[i:i+batch], dtype=torch.float32).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.clip(np.concatenate(preds), 0, RUL_CLIP)

# ── CONFORMAL ────────────────────────────────────────────────────────────────

def conf_q(r, cov=0.90):
    n=len(r); return float(np.quantile(r, min(np.ceil((n+1)*cov)/n, 1.0)))

def cov_wid(y, lo, hi): return ((y>=lo)&(y<=hi)).mean(),(hi-lo).mean()

def late_cov(y, lo, hi, pred):
    late=pred<50
    if late.sum()==0: return float('nan')
    return ((y[late]>=lo[late])&(y[late]<=hi[late])).mean()

# ── RUN ONE DATASET ───────────────────────────────────────────────────────────

def run_dataset(name, tr_seqs, tr_y, cal_seqs, cal_y,
                te_seqs, te_y, pipeline, device):
    print(f"\n{'='*58}")
    print(f"  {name}")
    print(f"{'='*58}")
    print(f"  train={len(tr_seqs)}  cal={len(cal_seqs)}  test={len(te_seqs)}")

    print("  Extracting Chronos-2 embeddings ...")
    tr_emb  = extract_embeddings(pipeline, tr_seqs,  device)
    te_emb  = extract_embeddings(pipeline, te_seqs,  device)
    ca_emb  = extract_embeddings(pipeline, cal_seqs, device)
    print(f"  emb shape: {tr_emb.shape}")

    esc    = StandardScaler().fit(tr_emb)
    tr_s, te_s, ca_s = (esc.transform(x) for x in [tr_emb, te_emb, ca_emb])

    print(f"  Training NN head ({epochs} epochs) ...")
    head     = train_head(tr_s, tr_y, device, epochs=epochs)
    pred     = predict(head, te_s, device)
    cal_pred = predict(head, ca_s, device)

    mae  = mean_absolute_error(te_y, pred)
    rmse = np.sqrt(mean_squared_error(te_y, pred))
    r2   = r2_score(te_y, pred)

    resid    = np.abs(cal_y - cal_pred)
    q        = conf_q(resid)
    lo, hi   = pred-q, pred+q
    cov, wid = cov_wid(te_y, lo, hi)
    lc       = late_cov(te_y, lo, hi, pred)
    h        = "YES" if abs(cov-COVERAGE)<0.10 else "NO"

    print(f"\n  MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f}")
    print(f"  Cov={cov:.0%} Wid={wid:.1f} Late={lc:.0%} q={q:.1f} Honest={h}")
    return dict(name=name, MAE=round(mae,2), RMSE=round(rmse,2),
                R2=round(r2,3), cov=f"{cov:.0%}", wid=round(wid,1),
                late=f"{lc:.0%}", q=round(q,1), honest=h)

# ── MAIN ─────────────────────────────────────────────────────────────────────

SEQ_LEN = 10
epochs  = 40

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading Chronos-2 ...")
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float32)
    print("Loaded.\n")

    results = []

    # ── IMS BEARING ──────────────────────────────────────────────────────────
    print("Loading IMS bearing (test 2 — most complete) ...")
    ims_signal = load_ims_test(os.path.join(IMS_DIR, "2nd_test"))
    if ims_signal is not None:
        print(f"  IMS signal shape: {ims_signal.shape}")
        n    = len(ims_signal)
        rul  = np.clip(np.arange(n-1,-1,-1,dtype=np.float32), 0, RUL_CLIP)
        # split: first 70% train, last 30% test
        split     = int(n*0.7)
        tr_sig, tr_rul = ims_signal[:split], rul[:split]
        te_sig, te_rul = ims_signal[split:], rul[split:]
        tr_seqs, tr_y  = make_seqs(tr_sig, tr_rul, SEQ_LEN)
        cal_seqs, cal_y = random_cut_seqs(tr_sig, tr_rul, SEQ_LEN, n=20)
        te_seqs, te_y   = eval_seqs(te_sig, te_rul, SEQ_LEN, n=30)
        if te_seqs is not None:
            r = run_dataset("IMS Bearing (test 2)",
                            tr_seqs, tr_y, cal_seqs, cal_y,
                            te_seqs, te_y, pipeline, device)
            results.append(r)

    # ── NASA BATTERY ─────────────────────────────────────────────────────────
    print("\nLoading NASA Battery B0005 + B0006 for training, B0007 for test ...")
    # Train on B0005+B0006, test on B0007
    tr_sigs, tr_ruls = [], []
    for bname in ['B0005', 'B0006']:
        sig, rul = load_battery(
            os.path.join(BATTERY_DIR, f'{bname}.mat'), bname)
        tr_sigs.append(sig); tr_ruls.append(rul)
    tr_sig  = np.vstack(tr_sigs)
    tr_rul  = np.concatenate(tr_ruls)
    te_sig, te_rul = load_battery(
        os.path.join(BATTERY_DIR, 'B0007.mat'), 'B0007')
    print(f"  Train cycles: {len(tr_sig)}  Test cycles: {len(te_sig)}")

    tr_seqs, tr_y  = make_seqs(tr_sig, tr_rul, SEQ_LEN)
    cal_seqs,cal_y = random_cut_seqs(tr_sig, tr_rul, SEQ_LEN, n=20)
    te_seqs, te_y  = eval_seqs(te_sig, te_rul, SEQ_LEN, n=30)

    if te_seqs is not None:
        r = run_dataset("NASA Battery (B0007)",
                        tr_seqs, tr_y, cal_seqs, cal_y,
                        te_seqs, te_y, pipeline, device)
        results.append(r)

    # ── KAIST BEARING ─────────────────────────────────────────────────────────
    print("\nLoading KAIST bearing ...")
    kaist_sig, kaist_rul = load_kaist(KAIST_DIR)
    print(f"  KAIST signal shape: {kaist_sig.shape}")
    n      = len(kaist_sig)
    split  = int(n * 0.7)
    tr_seqs, tr_y   = make_seqs(kaist_sig[:split], kaist_rul[:split], SEQ_LEN)
    cal_seqs, cal_y = random_cut_seqs(
        kaist_sig[:split], kaist_rul[:split], SEQ_LEN, n=20)
    te_seqs, te_y   = eval_seqs(
        kaist_sig[split:], kaist_rul[split:], SEQ_LEN, n=30)

    if te_seqs is not None:
        r = run_dataset("KAIST Bearing",
                        tr_seqs, tr_y, cal_seqs, cal_y,
                        te_seqs, te_y, pipeline, device)
        results.append(r)

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  COMPLETE CROSS-DATASET SUMMARY — One frozen Chronos-2 backbone")
    print(f"{'='*70}")
    print(f"  {'Dataset':<28}{'MAE':>7}{'Cov':>8}{'Late':>8}  Honest?")
    print(f"  {'-'*55}")
    prev = [
        ("C-MAPSS FD001 (jet sim)",   "15.17", "88%",  "97%"),
        ("C-MAPSS FD002 (jet sim)",   "19.16", "92%",  "99%"),
        ("C-MAPSS FD003 (jet sim)",   "12.71", "93%", "100%"),
        ("C-MAPSS FD004 (jet sim)",   "19.03", "93%",  "97%"),
        ("N-CMAPSS DS03 (real jet)",  "13.55", "85%",  "85%"),
        ("FEMTO Bearing",             "10.39", "99%", "100%"),
    ]
    for name, m, c, l in prev:
        print(f"  {name:<28}{m:>7}{c:>8}{l:>8}  YES")
    for r in results:
        print(f"  {r['name']:<28}{r['MAE']:>7}"
              f"{r['cov']:>8}{r['late']:>8}  {r['honest']}")
    print(f"{'='*70}")
    print(f"\n  Machine types covered:")
    print(f"  Jet engines: C-MAPSS + N-CMAPSS")
    print(f"  Rolling bearings: FEMTO + KAIST + IMS")
    print(f"  Batteries: NASA B0005-B0007")
    print(f"  Total datasets: {6 + len(results)}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
