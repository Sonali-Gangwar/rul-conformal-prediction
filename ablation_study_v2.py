"""
ablation_study_v2.py — Foundation Model Ablation Study (Fixed)
Compares patch size, depth, pre-training on FD001 and FD002
Uses the SAME training pipeline as your main experiment
Run: python ablation_study_v2.py
Time: ~2 hours on RTX 3060
"""

import os, sys, json, copy, traceback
print("Starting ablation study v2...")
sys.stdout.flush()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

DATA_DIR = r"E:\rul_project\data"
BACKBONE = r"E:\rul_project\iclr_backbone_v2.pt"
OUT_FILE = r"E:\rul_project\ablation_results_v2.json"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED     = 42
COLS     = ["engine","cycle","op1","op2","op3"]+[f"s{i}" for i in range(1,22)]
DROP     = {"s1","s5","s6","s10","s16","s18","s19"}
RUL_CLIP = 125
NF       = 14
SEQ_LEN  = 30

print(f"Device: {DEVICE}")
sys.stdout.flush()

# ── ORIGINAL MODEL (exact same as your main experiment) ───────────────────────
class PatchEmbedding(nn.Module):
    def __init__(self, nf, patch_len=5, d_model=128, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.n_patches = SEQ_LEN // patch_len
        self.norm = nn.LayerNorm(nf * patch_len)
        self.proj = nn.Linear(nf * patch_len, d_model)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.3)
        nn.init.zeros_(self.proj.bias)
    def forward(self, x):
        B, L, C = x.shape
        n = L // self.patch_len
        x = x[:, :n*self.patch_len, :]
        x = x.reshape(B, n, self.patch_len * C)
        return self.drop(self.proj(self.norm(x)))

class PatchTST_Ablation(nn.Module):
    def __init__(self, nf=NF, patch_len=5, n_layers=4,
                 d_model=128, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.n_patches = SEQ_LEN // patch_len
        self.d_model   = d_model
        self.patch_embed = PatchEmbedding(nf, patch_len, d_model, dropout)
        self.pos = nn.Parameter(
            torch.randn(1, self.n_patches, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout,
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, n_layers)
        self.rul_head = None

    def encode(self, x):
        B, L, C = x.shape
        n = L // self.patch_len
        out = self.transformer(
            self.patch_embed(x) +
            self.pos[:, :n, :])
        return out.mean(dim=1)

    def add_head(self):
        self.rul_head = nn.Sequential(
            nn.Linear(self.d_model, 64), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x):
        return self.rul_head(self.encode(x)).squeeze(1)

class LSTMModel(nn.Module):
    def __init__(self, nf=NF, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(nf, hidden, 2,
                            batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Linear(32, 1))
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:,-1,:]).squeeze(1)

# ── DATA (same as main experiment) ────────────────────────────────────────────
def load_all(fd):
    df = pd.read_csv(
        os.path.join(DATA_DIR, f"train_{fd}.txt"),
        sep=r"\s+", header=None).iloc[:,:26]
    df.columns = COLS
    fc = [c for c in COLS if c.startswith("s") and c not in DROP]
    mc = df.groupby("engine")["cycle"].transform("max")
    df["RUL"] = (mc - df["cycle"]).clip(upper=RUL_CLIP)
    sc = StandardScaler()
    X  = sc.fit_transform(df[fc].values).astype(np.float32)
    return X, df["engine"].values, df["RUL"].values, sc, fc

def make_seqs(X, y, engines):
    seqs, labels = [], []
    for e in np.unique(engines):
        idx = np.where(engines==e)[0]
        Xe, ye = X[idx], y[idx]
        for end in range(SEQ_LEN, len(Xe)+1):
            seqs.append(Xe[end-SEQ_LEN:end])
            labels.append(ye[end-1])
    return (np.array(seqs, dtype=np.float32),
            np.array(labels, dtype=np.float32))

def rand_cut_cal(X, y, engines, seed=42, n=5):
    rng = np.random.RandomState(seed)
    seqs, ys = [], []
    for e in np.unique(engines):
        idx = np.where(engines==e)[0]
        Xe, ye = X[idx], y[idx]
        ne = len(Xe)
        if ne < SEQ_LEN: continue
        for _ in range(n):
            t = rng.randint(SEQ_LEN, ne)
            seqs.append(Xe[t-SEQ_LEN:t])
            ys.append(ye[t-1])
    return (np.array(seqs, dtype=np.float32),
            np.array(ys, dtype=np.float32))

def last_windows(X, engines):
    out = []
    for e in np.unique(engines):
        idx = np.where(engines==e)[0]
        Xe  = X[idx]
        if len(Xe) >= SEQ_LEN:
            out.append(Xe[-SEQ_LEN:])
        else:
            pad = np.zeros(
                (SEQ_LEN-len(Xe), Xe.shape[1]), dtype=np.float32)
            out.append(np.vstack([pad, Xe]))
    return np.array(out, dtype=np.float32)

def conf_q(resids, alpha=0.10):
    n  = len(resids)
    lv = min(np.ceil((n+1)*(1-alpha))/n, 1.0)
    return float(np.quantile(resids, lv))

def get_cov(y, pred, q):
    return float(((y>=pred-q)&(y<=pred+q)).mean())

def train_model(model, X, y, epochs=40, lr=5e-4,
                batch=64, is_lstm=False):
    model.to(DEVICE)
    if not is_lstm:
        model.add_head()
    model.to(DEVICE)
    opt   = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=15, gamma=0.5)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(np.clip(X,-4,4), dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)),
        batch_size=batch, shuffle=True)
    lf = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = lf(model(xb), yb)
            if not torch.isnan(loss):
                loss.backward(); opt.step()
        sched.step()
    return model

def do_predict(model, X, batch=256):
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(
                np.clip(X[i:i+batch],-4,4),
                dtype=torch.float32).to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    return np.clip(
        np.concatenate(preds), 0, RUL_CLIP)

# ── LOAD BACKBONE ─────────────────────────────────────────────────────────────
def try_load_backbone(model, backbone_path):
    """Load backbone weights — match keys flexibly"""
    try:
        sd = torch.load(backbone_path,
                        map_location=DEVICE,
                        weights_only=False)
        md = model.state_dict()
        matched = {}
        for k, v in sd.items():
            if k in md and md[k].shape == v.shape:
                matched[k] = v
        if matched:
            md.update(matched)
            model.load_state_dict(md, strict=False)
            return len(matched)
        return 0
    except Exception as e:
        print(f"    (backbone load failed: {e})")
        return 0

# ── RUN ONE EXPERIMENT ────────────────────────────────────────────────────────
def run_one(name, fd, patch_len, n_layers,
            pretrained, is_lstm,
            X_train, y_train, engines_train,
            X_test_last, y_test,
            X_all, y_all, engines_all):

    print(f"  [{fd}] {name}...", end=" ", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    # Calibration from random cutoff on FULL training set
    Xc, yc = rand_cut_cal(X_all, y_all, engines_all, seed=SEED)

    if is_lstm:
        model = LSTMModel(nf=NF)
        model = train_model(model, X_train, y_train,
                            epochs=40, lr=1e-3, is_lstm=True)
        pc  = do_predict(model, Xc)
        pt  = do_predict(model, X_test_last)
    else:
        model = PatchTST_Ablation(
            nf=NF, patch_len=patch_len, n_layers=n_layers)
        if pretrained:
            n_loaded = try_load_backbone(model, BACKBONE)
        model = train_model(model, X_train, y_train,
                            epochs=40, lr=5e-4)
        pc  = do_predict(model, Xc)
        pt  = do_predict(model, X_test_last)

    q   = conf_q(np.abs(yc - pc))
    mae = mean_absolute_error(y_test, pt)
    cov = get_cov(y_test, pt, q)

    print(f"MAE={mae:.2f}  Cov={cov:.0%}  q={q:.1f}")
    sys.stdout.flush()
    return {'MAE': round(mae, 2),
            'coverage': round(cov*100, 1),
            'q': round(q, 1)}

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*65)
    print("  ABLATION STUDY v2 — Foundation Model Design Validation")
    print("="*65)

    experiments = [
        # name,               patch, layers, pretrain, lstm
        ("FM Patch-5 4L (OURS)",    5,  4, True,  False),
        ("No Pre-training",          5,  4, False, False),
        ("Patch-10 4L",              10, 4, True,  False),
        ("Patch-3 4L",               3,  4, True,  False),
        ("Patch-5 2-Layer",          5,  2, True,  False),
        ("Patch-5 6-Layer",          5,  6, True,  False),
        ("LSTM Baseline",            5,  4, False, True),
    ]

    all_results = {}

    for fd in ["FD001", "FD002"]:
        print(f"\n{'─'*50}")
        print(f"  Dataset: {fd}")
        print(f"{'─'*50}")

        # Load full training data
        X_all, engines_all, rul_all, sc, fc = load_all(fd)

        # Full sequence training set
        X_train, y_train = make_seqs(X_all, rul_all, engines_all)
        print(f"  Training sequences: {len(X_train):,}")

        # Test set
        df_te = pd.read_csv(
            os.path.join(DATA_DIR, f"test_{fd}.txt"),
            sep=r"\s+", header=None).iloc[:,:26]
        df_te.columns = COLS
        Xte_raw = sc.transform(
            df_te[fc].values).astype(np.float32)
        y_test  = np.clip(pd.read_csv(
            os.path.join(DATA_DIR, f"RUL_{fd}.txt"),
            header=None).iloc[:,0].values, 0, RUL_CLIP)
        X_test_last = last_windows(Xte_raw, df_te["engine"].values)
        print(f"  Test engines: {len(np.unique(df_te['engine'])):,}")

        fd_results = {}
        for name, pl, nl, pt, lstm in experiments:
            try:
                r = run_one(
                    name, fd, pl, nl, pt, lstm,
                    X_train, y_train, engines_all,
                    X_test_last, y_test,
                    X_all, rul_all, engines_all)
                fd_results[name] = r
            except Exception as e:
                print(f"  ERROR in {name}: {e}")
                traceback.print_exc()
        all_results[fd] = fd_results

    # ── FINAL TABLE ──────────────────────────────────────────────────────────
    print("\n\n" + "="*72)
    print("  FINAL ABLATION TABLE")
    print("="*72)
    header = (f"  {'Variant':<28} {'FD001 MAE':>10} "
              f"{'FD001 Cov':>10} {'FD002 MAE':>10} {'FD002 Cov':>10}")
    print(header)
    print(f"  {'-'*68}")

    for name, _, _, _, _ in experiments:
        r1 = all_results.get("FD001",{}).get(name,{})
        r2 = all_results.get("FD002",{}).get(name,{})
        m1 = r1.get('MAE','—')
        c1 = f"{r1.get('coverage','—')}%" if 'coverage' in r1 else '—'
        m2 = r2.get('MAE','—')
        c2 = f"{r2.get('coverage','—')}%" if 'coverage' in r2 else '—'
        flag = " ← OURS" if "OURS" in name else ""
        print(f"  {name:<28} {str(m1):>10} {c1:>10} "
              f"{str(m2):>10} {c2:>10}{flag}")

    print("="*72)
    print("\n  What the results prove:")
    fd1 = all_results.get("FD001",{})
    ours_mae = fd1.get("FM Patch-5 4L (OURS)",{}).get('MAE',0)
    nopt_mae = fd1.get("No Pre-training",{}).get('MAE',0)
    lstm_mae = fd1.get("LSTM Baseline",{}).get('MAE',0)
    if ours_mae and nopt_mae:
        diff = round(nopt_mae - ours_mae, 2)
        print(f"  Pre-training benefit on FD001: "
              f"+{diff} MAE improvement")
    if ours_mae and lstm_mae:
        diff2 = round(lstm_mae - ours_mae, 2)
        print(f"  FM vs LSTM on FD001: "
              f"+{diff2} MAE improvement")

    with open(OUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {OUT_FILE}")
    print("  Done!")

if __name__ == "__main__":
    main()
