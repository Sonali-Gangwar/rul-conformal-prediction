"""
multiseed_exp.py — Multi-seed evaluation for foundation model paper

Runs PatchTST FM vs Scratch vs LSTM across 5 seeds on FD001-FD004
Reports mean ± std for MAE and coverage

Run: python multiseed_exp.py
Expected time: ~2 hours on RTX 3060
"""

import os, copy, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

DATA_DIR = r"E:\rul_project\data"
BACKBONE  = r"E:\rul_project\iclr_backbone_v2.pt"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN  = 30
PATCH_LEN = 5
N_PATCHES = 6
RUL_CLIP  = 125
NF        = 14
SEEDS     = [42, 7, 13, 21, 99]  # 5 seeds
COLS      = ["engine","cycle","op1","op2","op3"]+[f"s{i}" for i in range(1,22)]
DROP      = {"s1","s5","s6","s10","s16","s18","s19"}

print(f"Device: {DEVICE}")
print(f"Seeds: {SEEDS}")
print("="*65)
print("  Multi-seed Foundation Model Evaluation")
print("="*65)

# ── MODEL ─────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, nf, d_model=128, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(nf * PATCH_LEN)
        self.proj = nn.Linear(nf * PATCH_LEN, d_model)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.3)
        nn.init.zeros_(self.proj.bias)
    def forward(self, x):
        B, L, C = x.shape
        x = x.reshape(B, N_PATCHES, PATCH_LEN * C)
        return self.drop(self.proj(self.norm(x)))

class PatchTST_RUL(nn.Module):
    def __init__(self, nf=NF, d_model=128, n_heads=4,
                 n_layers=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.patch_embed = PatchEmbedding(nf, d_model, dropout)
        self.pos = nn.Parameter(torch.randn(1, N_PATCHES, d_model)*0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                          batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, n_layers)
        self.rul_head = None
    def encode(self, x):
        return self.transformer(self.patch_embed(x)+self.pos).mean(dim=1)
    def add_head(self):
        self.rul_head = nn.Sequential(
            nn.Linear(self.d_model, 64), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(64, 1))
    def forward_rul(self, x):
        return self.rul_head(self.encode(x)).squeeze(1)

class LSTMModel(nn.Module):
    def __init__(self, nf=NF, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(nf, hidden, 2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:,-1,:]).squeeze(1)

# ── DATA ──────────────────────────────────────────────────────────────────────

def load_fd(fd):
    df = pd.read_csv(os.path.join(DATA_DIR, f"train_{fd}.txt"),
                     sep=r"\s+", header=None).iloc[:,:26]
    df.columns = COLS
    fc = [c for c in COLS if c.startswith("s") and c not in DROP]
    mc = df.groupby("engine")["cycle"].transform("max")
    df["RUL"] = (mc - df["cycle"]).clip(upper=RUL_CLIP)
    sc = StandardScaler()
    X = sc.fit_transform(df[fc].values).astype(np.float32)
    return X, df["engine"].values, df["cycle"].values, df["RUL"].values, sc

def load_fd_test(fd, sc):
    df = pd.read_csv(os.path.join(DATA_DIR, f"test_{fd}.txt"),
                     sep=r"\s+", header=None).iloc[:,:26]
    df.columns = COLS
    fc = [c for c in COLS if c.startswith("s") and c not in DROP]
    X = sc.transform(df[fc].values).astype(np.float32)
    y = np.clip(pd.read_csv(os.path.join(DATA_DIR, f"RUL_{fd}.txt"),
                header=None).iloc[:,0].values, 0, RUL_CLIP)
    return X, df["engine"].values, df["cycle"].values, y

def make_seqs(X, y, engines):
    seqs, labels = [], []
    for e in np.unique(engines):
        idx = np.where(engines==e)[0]
        Xe, ye = X[idx], y[idx]
        for end in range(SEQ_LEN, len(Xe)+1):
            seqs.append(Xe[end-SEQ_LEN:end])
            labels.append(ye[end-1])
    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)

def last_wins(X, engines, cycles):
    out = []
    for e in np.unique(engines):
        idx = np.where(engines==e)[0]
        idx = idx[np.argsort(cycles[engines==e])]
        Xe = X[idx]
        if len(Xe) >= SEQ_LEN:
            out.append(Xe[-SEQ_LEN:])
        else:
            out.append(np.vstack([np.zeros((SEQ_LEN-len(Xe), NF)), Xe]))
    return np.array(out, dtype=np.float32)

def rand_cut(X, y, engines, cycles, seed=42, n_cuts=5):
    all_s, all_y = [], []
    for s in range(n_cuts):
        rng = np.random.RandomState(seed + s)
        for e in np.unique(engines):
            idx = np.where(engines==e)[0]
            idx = idx[np.argsort(cycles[engines==e])]
            Xe, ye = X[idx], y[idx]
            n = len(Xe)
            if n < SEQ_LEN: continue
            end = rng.randint(SEQ_LEN, n)
            all_s.append(Xe[end-SEQ_LEN:end])
            all_y.append(ye[end-1])
    return np.array(all_s, dtype=np.float32), np.array(all_y, dtype=np.float32)

def train_model(model, X, y, epochs=40, batch=64, lr=5e-4, is_lstm=False):
    model.to(DEVICE)
    if not is_lstm: model.add_head()
    model.to(DEVICE)
    loader = DataLoader(TensorDataset(
        torch.tensor(np.clip(X,-4,4), dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)),
        batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
    lf = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb) if is_lstm else model.forward_rul(xb)
            loss = lf(pred, yb)
            if not torch.isnan(loss):
                loss.backward(); opt.step()
        sched.step()
    return model

def predict(model, X, is_lstm=False, batch=256):
    model.eval(); preds = []
    Xt = torch.tensor(np.clip(X,-4,4), dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(Xt), batch):
            xb = Xt[i:i+batch].to(DEVICE)
            p = (model(xb) if is_lstm
                 else model.forward_rul(xb)).cpu().numpy()
            preds.append(np.nan_to_num(p, nan=62.5))
    return np.clip(np.concatenate(preds), 0, RUL_CLIP)

def conf_q(r, cov=0.90):
    n = len(r)
    return float(np.quantile(r, min(np.ceil((n+1)*cov)/n, 1.0)))

def get_cov(y, pred, q):
    return float(((y >= pred-q) & (y <= pred+q)).mean())

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_one_seed(fd, seed, backbone):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, engines, cycles, rul, sc = load_fd(fd)
    Xt, et, ct, y_test = load_fd_test(fd, sc)

    all_e = np.unique(engines)
    rng = np.random.RandomState(seed)
    rng.shuffle(all_e)
    cal_e = set(all_e[-20:])
    mtr = ~np.isin(engines, list(cal_e))
    mca =  np.isin(engines, list(cal_e))

    Xs, ys = make_seqs(X[mtr], rul[mtr], engines[mtr])
    cal_seqs, cal_y = rand_cut(X[mca], rul[mca], engines[mca],
                                cycles[mca], seed=seed)
    te_seqs = last_wins(Xt, et, ct)

    results = {}

    # FM
    fm = copy.deepcopy(backbone)
    fm = train_model(fm, Xs, ys, epochs=40)
    p_fm = predict(fm, te_seqs)
    cp_fm = predict(fm, cal_seqs)
    q_fm = conf_q(np.abs(cal_y - cp_fm))
    results['FM'] = {
        'MAE': round(mean_absolute_error(y_test, p_fm), 2),
        'cov': round(get_cov(y_test, p_fm, q_fm), 3)
    }

    # Scratch
    sc_m = PatchTST_RUL().to(DEVICE)
    sc_m = train_model(sc_m, Xs, ys, epochs=40)
    p_sc = predict(sc_m, te_seqs)
    cp_sc = predict(sc_m, cal_seqs)
    q_sc = conf_q(np.abs(cal_y - cp_sc))
    results['Scratch'] = {
        'MAE': round(mean_absolute_error(y_test, p_sc), 2),
        'cov': round(get_cov(y_test, p_sc, q_sc), 3)
    }

    # LSTM
    lstm = LSTMModel().to(DEVICE)
    lstm = train_model(lstm, Xs, ys, epochs=40, lr=1e-3, is_lstm=True)
    p_ls = predict(lstm, te_seqs, is_lstm=True)
    cp_ls = predict(lstm, cal_seqs, is_lstm=True)
    q_ls = conf_q(np.abs(cal_y - cp_ls))
    results['LSTM'] = {
        'MAE': round(mean_absolute_error(y_test, p_ls), 2),
        'cov': round(get_cov(y_test, p_ls, q_ls), 3)
    }

    return results

def main():
    # Load backbone once
    print("\n[STEP 1] Loading pre-trained backbone...")
    backbone = PatchTST_RUL().to(DEVICE)
    backbone.load_state_dict(torch.load(
        BACKBONE, map_location=DEVICE, weights_only=False), strict=False)
    print("  Loaded: iclr_backbone_v2.pt")

    all_results = {}

    for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
        print(f"\n{'='*65}")
        print(f"  Dataset: {fd}")
        print(f"{'='*65}")
        fd_results = {'FM': [], 'Scratch': [], 'LSTM': []}

        for seed in SEEDS:
            print(f"  Seed {seed}...", end=' ', flush=True)
            r = run_one_seed(fd, seed, backbone)
            for model in ['FM', 'Scratch', 'LSTM']:
                fd_results[model].append(r[model])
            print(f"FM MAE={r['FM']['MAE']} Cov={r['FM']['cov']:.0%}")

        # Compute mean ± std
        print(f"\n  Results for {fd}:")
        print(f"  {'Method':<12}{'MAE mean':>10}{'MAE std':>10}"
              f"{'Cov mean':>10}{'Cov std':>10}")
        print(f"  {'-'*52}")

        fd_summary = {}
        for model in ['FM', 'Scratch', 'LSTM']:
            maes = [r['MAE'] for r in fd_results[model]]
            covs = [r['cov'] for r in fd_results[model]]
            mae_m = round(np.mean(maes), 2)
            mae_s = round(np.std(maes), 2)
            cov_m = round(np.mean(covs)*100, 1)
            cov_s = round(np.std(covs)*100, 1)
            fd_summary[model] = {
                'MAE_mean': mae_m, 'MAE_std': mae_s,
                'cov_mean': cov_m, 'cov_std': cov_s
            }
            print(f"  {model:<12}{mae_m:>10}{mae_s:>10}"
                  f"{cov_m:>9.1f}%{cov_s:>9.1f}%")

        all_results[fd] = fd_summary

    # Final table
    print(f"\n\n{'='*75}")
    print("  FINAL TABLE — Mean ± Std across 5 seeds")
    print(f"{'='*75}")
    print(f"  {'FD':<6}{'FM MAE':>12}{'FM Cov':>10}"
          f"{'Scratch MAE':>14}{'Scratch Cov':>13}"
          f"{'LSTM MAE':>12}{'LSTM Cov':>10}")
    print(f"  {'-'*70}")
    for fd, r in all_results.items():
        fm = r['FM']; sc = r['Scratch']; ls = r['LSTM']
        print(f"  {fd:<6}"
              f"{fm['MAE_mean']}±{fm['MAE_std']:>5}{fm['cov_mean']}±{fm['cov_std']:>4}%"
              f"{sc['MAE_mean']:>8}±{sc['MAE_std']:<5}{sc['cov_mean']:>8}±{sc['cov_std']:<4}%"
              f"{ls['MAE_mean']:>8}±{ls['MAE_std']:<5}{ls['cov_mean']:>8}±{ls['cov_std']:<4}%")
    print(f"{'='*75}")

    with open(r'E:\rul_project\multiseed_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("  Saved: multiseed_results.json")

if __name__ == "__main__":
    main()
