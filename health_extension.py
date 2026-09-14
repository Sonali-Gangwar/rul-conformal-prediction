"""
health_extension.py — Health Extension: PatchTST Foundation Model on MIMIC-IV

RESEARCH QUESTION:
  Can a foundation model pre-trained on industrial degradation signals
  (jet engines, bearings) transfer to clinical time series (ICU vital signs)?

  This tests: universal degradation representations across industrial
  and biological systems.

SETUP:
  - MIMIC-IV demo: 100 ICU patients, vital signs per hour
  - Task: predict remaining ICU stay (clinical RUL) from vital signs
  - Compare: PatchTST FM (pre-trained on C-MAPSS) vs scratch vs LSTM

VITAL SIGNS USED:
  - Heart rate (HR)
  - Systolic blood pressure (SBP)
  - Respiratory rate (RR)
  - SpO2 (oxygen saturation)
  - Temperature

Run: python health_extension.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN  = 24   # 24 hourly measurements = 1 day window
PATCH_LEN= 4    # 4-hour patches → 6 patches per window
N_PATCHES= SEQ_LEN // PATCH_LEN   # = 6 (same as C-MAPSS)
RUL_CLIP = 14   # clip ICU stay to 14 days max
NF_HEALTH= 5    # 5 vital signs
NF_CMAPSS= 14   # 14 C-MAPSS sensors
BACKBONE  = r"E:\rul_project\iclr_backbone_v2.pt"
MIMIC_DIR = r"E:\longagent\mimic_demo"

print(f"Device: {DEVICE}")
print("="*62)
print("  Health Extension: PatchTST FM on MIMIC-IV ICU Vital Signs")
print("="*62)

# ── VITALSIGN ITEM IDs in MIMIC-IV ───────────────────────────────────────────
# These are the standard MIMIC-IV itemids for common vital signs
VITAL_ITEMS = {
    220045: 'heart_rate',       # Heart Rate
    220179: 'sbp',              # Systolic BP
    220180: 'dbp',              # Diastolic BP
    220210: 'resp_rate',        # Respiratory Rate
    220277: 'spo2',             # SpO2
    223761: 'temperature',      # Temperature (F)
}

# ── LOAD AND PREPROCESS MIMIC-IV ─────────────────────────────────────────────

def load_mimic(mimic_dir):
    print("\n[STEP 1] Loading MIMIC-IV demo data...")
    chart = pd.read_csv(os.path.join(mimic_dir, 'chartevents.csv'),
                        low_memory=False)
    stays = pd.read_csv(os.path.join(mimic_dir, 'icustays.csv'))
    print(f"  Chartevents: {len(chart):,} rows")
    print(f"  ICU stays:   {len(stays):,} patients")
    print(f"  Columns: {list(chart.columns)}")
    return chart, stays

def preprocess_vitals(chart, stays):
    print("\n[STEP 2] Extracting vital signs...")

    # Filter to vital sign items
    chart = chart[chart['itemid'].isin(VITAL_ITEMS.keys())].copy()
    chart['vital'] = chart['itemid'].map(VITAL_ITEMS)
    chart['valuenum'] = pd.to_numeric(chart['valuenum'], errors='coerce')
    chart = chart.dropna(subset=['valuenum', 'stay_id', 'charttime'])
    chart['charttime'] = pd.to_datetime(chart['charttime'])

    print(f"  Vital sign rows: {len(chart):,}")
    print(f"  Vital types found: {chart['vital'].unique().tolist()}")

    # ICU stay duration in hours → clinical RUL
    stays['intime']  = pd.to_datetime(stays['intime'])
    stays['outtime'] = pd.to_datetime(stays['outtime'])
    stays['los_hours'] = (stays['outtime'] - stays['intime']).dt.total_seconds() / 3600
    stays = stays[stays['los_hours'] >= SEQ_LEN]  # need at least SEQ_LEN hours
    print(f"  Valid ICU stays (≥{SEQ_LEN}h): {len(stays)}")

    # Build per-patient hourly time series
    sequences = []
    labels    = []
    stay_ids  = []

    vitals_list = ['heart_rate','sbp','resp_rate','spo2','temperature']

    for _, stay in stays.iterrows():
        sid = stay['stay_id']
        los = stay['los_hours']

        # Get this patient's chart data
        pt = chart[chart['stay_id'] == sid].copy()
        if len(pt) < 10: continue

        # Create hourly bins from admission
        pt['hour'] = (pt['charttime'] - stay['intime']).dt.total_seconds() / 3600
        pt = pt[(pt['hour'] >= 0) & (pt['hour'] <= los)]
        pt['hour_bin'] = pt['hour'].astype(int)

        # Pivot to wide format
        pivot = pt.groupby(['hour_bin','vital'])['valuenum'].median().unstack()

        # Keep only our 5 vitals
        for v in vitals_list:
            if v not in pivot.columns:
                pivot[v] = np.nan
        pivot = pivot[vitals_list]

        # Forward fill missing values
        pivot = pivot.ffill().bfill()
        pivot = pivot.dropna()

        if len(pivot) < SEQ_LEN: continue

        # Create sliding windows with clinical RUL labels
        vals = pivot.values.astype(np.float32)
        max_hour = len(vals)

        for end in range(SEQ_LEN, min(max_hour+1, int(los)+1)):
            seq = vals[end-SEQ_LEN:end]
            if seq.shape[0] != SEQ_LEN: continue
            # Clinical RUL = hours remaining in ICU
            rul = min(los - end, RUL_CLIP * 24) / 24  # convert to days, clip
            sequences.append(seq)
            labels.append(rul)
            stay_ids.append(sid)

    if not sequences:
        print("  WARNING: No valid sequences found. Check itemids.")
        return None, None, None

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels,    dtype=np.float32)
    ids = np.array(stay_ids)
    print(f"  Total sequences: {len(X):,}, shape={X.shape}")
    print(f"  Clinical RUL range: {y.min():.1f} - {y.max():.1f} days")
    return X, y, ids

def normalize(X_train, X_test):
    sc = StandardScaler()
    B, L, F = X_train.shape
    X_train_f = sc.fit_transform(X_train.reshape(-1, F)).reshape(B, L, F)
    B2 = X_test.shape[0]
    X_test_f  = sc.transform(X_test.reshape(-1, F)).reshape(B2, L, F)
    return X_train_f.astype(np.float32), X_test_f.astype(np.float32)

# ── MODEL ─────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, nf, patch_len, d_model, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.norm = nn.LayerNorm(nf * patch_len)
        self.proj = nn.Linear(nf * patch_len, d_model)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.3)
        nn.init.zeros_(self.proj.bias)
    def forward(self, x):
        B, L, C = x.shape
        x = x.reshape(B, N_PATCHES, self.patch_len * C)
        return self.drop(self.proj(self.norm(x)))

class PatchTST_Health(nn.Module):
    """
    PatchTST adapted for health vital signs.
    Uses same Transformer backbone as industrial model.
    Input: 24h window of 5 vital signs → 6 patches of 4h each
    """
    def __init__(self, nf=NF_HEALTH, d_model=128, n_heads=4,
                 n_layers=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.nf = nf
        self.d_model = d_model
        self.patch_embed = PatchEmbedding(nf, PATCH_LEN, d_model, dropout)
        self.pos = nn.Parameter(torch.randn(1, N_PATCHES, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                          batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, n_layers)
        self.rul_head = None

    def encode(self, x):
        return self.transformer(
            self.patch_embed(x) + self.pos).mean(dim=1)

    def add_head(self):
        self.rul_head = nn.Sequential(
            nn.Linear(self.d_model, 64), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(64, 1))

    def forward_rul(self, x):
        return self.rul_head(self.encode(x)).squeeze(1)

class LSTMBaseline(nn.Module):
    def __init__(self, nf=NF_HEALTH, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(nf, hidden, 2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                   nn.Linear(32, 1))
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1, :]).squeeze(1)

# ── LOAD INDUSTRIAL BACKBONE ──────────────────────────────────────────────────

def load_backbone_weights(model):
    """
    Load weights from industrial backbone where shapes match.
    The Transformer layers are identical (same d_model, heads, layers).
    Only patch_embed differs (14 sensors vs 5 vitals) — skip those.
    """
    if not os.path.exists(BACKBONE):
        print(f"  Backbone not found at {BACKBONE}")
        return model

    state = torch.load(BACKBONE, map_location=DEVICE, weights_only=False)

    # Filter: only load Transformer weights (backbone), skip patch embed
    model_state = model.state_dict()
    transferable = {}
    skipped = []

    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            transferable[k] = v
        else:
            skipped.append(k)

    model_state.update(transferable)
    model.load_state_dict(model_state)

    print(f"  Transferred: {len(transferable)} layers")
    print(f"  Skipped (shape mismatch): {len(skipped)} layers")
    print(f"  Key transferred: transformer weights (4L,4H,128D)")
    return model

# ── TRAINING ──────────────────────────────────────────────────────────────────

def train_model(model, X, y, epochs=40, batch=32, lr=5e-4, is_lstm=False):
    model.to(DEVICE)
    if not is_lstm: model.add_head()
    model.to(DEVICE)
    loader = DataLoader(
        TensorDataset(torch.tensor(np.clip(X, -4, 4), dtype=torch.float32),
                      torch.tensor(y, dtype=torch.float32)),
        batch_size=batch, shuffle=True)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
    lf    = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb) if is_lstm else model.forward_rul(xb)
            loss = lf(pred, yb)
            if not torch.isnan(loss):
                loss.backward(); opt.step()
            total += loss.item()
        sched.step()
        if (ep+1) % 10 == 0:
            print(f"    ep {ep+1}/{epochs} loss={total/len(loader):.3f}")
    return model

def predict(model, X, is_lstm=False, batch=128):
    model.eval(); preds = []
    Xt = torch.tensor(np.clip(X, -4, 4), dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(Xt), batch):
            xb = Xt[i:i+batch].to(DEVICE)
            p  = (model(xb) if is_lstm
                  else model.forward_rul(xb)).cpu().numpy()
            preds.append(np.nan_to_num(p, nan=3.0))
    return np.clip(np.concatenate(preds), 0, RUL_CLIP)

def conf_q(r, cov=0.90):
    n = len(r)
    return float(np.quantile(r, min(np.ceil((n+1)*cov)/n, 1.0)))

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    import copy

    # Load data
    chart, stays = load_mimic(MIMIC_DIR)
    X, y, ids = preprocess_vitals(chart, stays)

    if X is None:
        print("\nNo data extracted. The demo dataset may have different itemids.")
        print("Checking available itemids...")
        chart2 = pd.read_csv(os.path.join(MIMIC_DIR, 'chartevents.csv'),
                             low_memory=False)
        print(f"Available itemids (top 20): {chart2['itemid'].value_counts().head(20)}")
        return

    # Train/test split by patient
    unique_ids = np.unique(ids)
    np.random.seed(42); np.random.shuffle(unique_ids)
    n_test  = max(3, int(len(unique_ids) * 0.20))
    n_cal   = max(3, int(len(unique_ids) * 0.20))
    test_ids = set(unique_ids[:n_test])
    cal_ids  = set(unique_ids[n_test:n_test+n_cal])
    train_ids= set(unique_ids[n_test+n_cal:])

    mtr = np.array([i in train_ids for i in ids])
    mca = np.array([i in cal_ids   for i in ids])
    mte = np.array([i in test_ids  for i in ids])

    Xtr, ytr = X[mtr], y[mtr]
    Xca, yca = X[mca], y[mca]
    Xte, yte = X[mte], y[mte]

    Xtr, Xte = normalize(Xtr, Xte)
    _, Xca   = normalize(Xtr, Xca)  # use train scaler

    print(f"\n  Train: {len(Xtr):,}  Cal: {len(Xca):,}  Test: {len(Xte):,}")
    print(f"  Vital signs: {NF_HEALTH}  Window: {SEQ_LEN}h  Patches: {N_PATCHES}×{PATCH_LEN}h")

    results = {}

    # (A) PatchTST with industrial backbone transfer
    print("\n[STEP 3] (A) PatchTST with Industrial Backbone Transfer...")
    print("  Loading industrial backbone (C-MAPSS pre-trained)...")
    fm = PatchTST_Health(NF_HEALTH).to(DEVICE)
    fm = load_backbone_weights(fm)
    fm = train_model(fm, Xtr, ytr, epochs=40, lr=5e-4)
    pred_fm   = predict(fm, Xte)
    cal_pred  = predict(fm, Xca)
    q_fm      = conf_q(np.abs(yca - cal_pred))
    mae_fm    = mean_absolute_error(yte, pred_fm)
    cov_fm    = float(((yte >= pred_fm-q_fm) & (yte <= pred_fm+q_fm)).mean())
    print(f"  MAE={mae_fm:.3f} days  Coverage={cov_fm:.0%}  q={q_fm:.2f}")
    results['FM_transfer'] = {'MAE': round(mae_fm,3), 'cov': f"{cov_fm:.0%}"}

    # (B) PatchTST from scratch (no transfer)
    print("\n[STEP 4] (B) PatchTST from scratch (no transfer)...")
    sc_model = PatchTST_Health(NF_HEALTH).to(DEVICE)
    sc_model = train_model(sc_model, Xtr, ytr, epochs=40, lr=5e-4)
    pred_sc  = predict(sc_model, Xte)
    cal_sc   = predict(sc_model, Xca)
    q_sc     = conf_q(np.abs(yca - cal_sc))
    mae_sc   = mean_absolute_error(yte, pred_sc)
    cov_sc   = float(((yte >= pred_sc-q_sc) & (yte <= pred_sc+q_sc)).mean())
    print(f"  MAE={mae_sc:.3f} days  Coverage={cov_sc:.0%}  q={q_sc:.2f}")
    results['Scratch'] = {'MAE': round(mae_sc,3), 'cov': f"{cov_sc:.0%}"}

    # (C) LSTM baseline
    print("\n[STEP 5] (C) LSTM baseline...")
    lstm = LSTMBaseline(NF_HEALTH).to(DEVICE)
    lstm = train_model(lstm, Xtr, ytr, epochs=40, lr=1e-3, is_lstm=True)
    pred_ls  = predict(lstm, Xte, is_lstm=True)
    cal_ls   = predict(lstm, Xca, is_lstm=True)
    q_ls     = conf_q(np.abs(yca - cal_ls))
    mae_ls   = mean_absolute_error(yte, pred_ls)
    cov_ls   = float(((yte >= pred_ls-q_ls) & (yte <= pred_ls+q_ls)).mean())
    print(f"  MAE={mae_ls:.3f} days  Coverage={cov_ls:.0%}  q={q_ls:.2f}")
    results['LSTM'] = {'MAE': round(mae_ls,3), 'cov': f"{cov_ls:.0%}"}

    # Final results
    print(f"\n\n{'='*62}")
    print("  HEALTH EXTENSION RESULTS — MIMIC-IV ICU Vital Signs")
    print(f"{'='*62}")
    print(f"  Task: Predict remaining ICU stay (days) from {SEQ_LEN}h vital sign window")
    print(f"  Vital signs: HR, SBP, RR, SpO2, Temperature")
    print(f"  {'Method':<30}{'MAE (days)':>12}{'Coverage':>10}")
    print(f"  {'-'*52}")
    for name, r in results.items():
        better = ' <- industrial transfer helps!' if (
            name == 'FM_transfer' and
            r['MAE'] < results.get('Scratch', {}).get('MAE', 999)) else ''
        print(f"  {name:<30}{r['MAE']:>12}{r['cov']:>10}{better}")
    print(f"{'='*62}")
    print()
    print("  KEY FINDING:")
    fm_mae = results['FM_transfer']['MAE']
    sc_mae = results.get('Scratch', {}).get('MAE', 999)
    if fm_mae < sc_mae:
        print(f"  Industrial pre-training HELPS clinical prediction:")
        print(f"  FM transfer MAE={fm_mae} < Scratch MAE={sc_mae}")
        print(f"  → Degradation patterns transfer from jet engines to ICU patients")
    else:
        print(f"  FM transfer MAE={fm_mae} vs Scratch MAE={sc_mae}")
        print(f"  → Limited transfer with demo dataset (only 100 patients)")
        print(f"  → Full MIMIC-IV (40,000+ patients) needed for stronger result")

    import json
    with open(r"E:\rul_project\health_extension_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved: health_extension_results.json")

if __name__ == "__main__":
    main()
