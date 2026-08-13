"""
train_baseline.py  —  Step 1 of the RUL project (FD001)

Trains the three baseline models (Random Forest, XGBoost, LSTM) and prints a
metrics table (MAE, RMSE, R2, training time) to compare with Ayushi's slides.

It also saves, for the LSTM:
  - calibration residuals  -> outputs/calib_residuals.npy   (needed for Step 2: conformal)
  - test predictions       -> outputs/test_predictions.npz

Run:  python train_baseline.py
"""

import os
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

OUT_DIR = "outputs"
SEQ_LEN = 30          # LSTM looks at the last 30 cycles
RUL_CLIP = 125


# ----------------------------- helpers -----------------------------
def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def last_cycle_features(test_X, test_engine, test_cycle):
    """For RF/XGB: take the feature row at each test engine's LAST cycle."""
    feats = []
    for eng in np.unique(test_engine):
        mask = test_engine == eng
        idx = np.where(mask)[0]
        last_idx = idx[np.argmax(test_cycle[mask])]
        feats.append(test_X[last_idx])
    return np.array(feats)


def make_sequences(X, y, engine, seq_len):
    """Build (samples, seq_len, features) windows for the LSTM from training data."""
    seqs, labels = [], []
    for eng in np.unique(engine):
        idx = np.where(engine == eng)[0]
        Xe, ye = X[idx], y[idx]
        for end in range(seq_len, len(Xe) + 1):
            seqs.append(Xe[end - seq_len:end])
            labels.append(ye[end - 1])
    return np.array(seqs), np.array(labels)


def last_sequences(test_X, test_engine, test_cycle, seq_len, n_features):
    """For each test engine, build the sequence ending at its last cycle (pad if short)."""
    seqs = []
    for eng in np.unique(test_engine):
        idx = np.where(test_engine == eng)[0]
        idx = idx[np.argsort(test_cycle[test_engine == eng])]
        Xe = test_X[idx]
        if len(Xe) >= seq_len:
            seqs.append(Xe[-seq_len:])
        else:
            pad = np.zeros((seq_len - len(Xe), n_features))
            seqs.append(np.vstack([pad, Xe]))
    return np.array(seqs)


# ----------------------------- main -----------------------------
def main():
    data = np.load(os.path.join(OUT_DIR, "prepared.npz"), allow_pickle=True)
    train_X, train_y = data["train_X"], data["train_y"]
    train_engine, train_cycle = data["train_engine"], data["train_cycle"]
    test_X, test_engine, test_cycle = data["test_X"], data["test_engine"], data["test_cycle"]
    true_rul = data["true_rul"]
    n_features = train_X.shape[1]

    # test targets for RF/XGB/LSTM = the provided true RUL (capped to match training)
    y_test = np.clip(true_rul, 0, RUL_CLIP)

    results = {}

    # ---------- Random Forest ----------
    test_feats = last_cycle_features(test_X, test_engine, test_cycle)
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(train_X, train_y)
    rf_pred = rf.predict(test_feats)
    results["Random Forest"] = (*metrics(y_test, rf_pred), time.time() - t0)

    # ---------- XGBoost ----------
    t0 = time.time()
    xg = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                          random_state=42, n_jobs=-1)
    xg.fit(train_X, train_y)
    xg_pred = xg.predict(test_feats)
    results["XGBoost"] = (*metrics(y_test, xg_pred), time.time() - t0)

    # ---------- LSTM ----------
    lstm_pred, calib_resid = train_lstm(
        train_X, train_y, train_engine, test_X, test_engine, test_cycle,
        y_test, n_features, results)

    # ---------- print table ----------
    print("\n" + "=" * 60)
    print(f"{'Model':<16}{'MAE':>8}{'RMSE':>8}{'R2':>8}{'Time(s)':>10}")
    print("-" * 60)
    for name, (mae, rmse, r2, t) in results.items():
        print(f"{name:<16}{mae:>8.2f}{rmse:>8.2f}{r2:>8.3f}{t:>10.1f}")
    print("=" * 60)

    # ---------- save artefacts for Step 2 (conformal) ----------
    np.save(os.path.join(OUT_DIR, "calib_residuals.npy"), calib_resid)
    np.savez(os.path.join(OUT_DIR, "test_predictions.npz"),
             rf=rf_pred, xgb=xg_pred, lstm=lstm_pred, y_test=y_test)
    print("\nSaved calibration residuals + test predictions for Step 2 (conformal).")


def train_lstm(train_X, train_y, train_engine, test_X, test_engine, test_cycle,
               y_test, n_features, results):
    """Train a small LSTM. Returns (test_predictions, calibration_residuals)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"LSTM training on: {device}")

    Xs, ys = make_sequences(train_X, train_y, train_engine, SEQ_LEN)

    # split a CALIBRATION set out of training — this is what conformal needs later
    X_tr, X_cal, y_tr, y_cal = train_test_split(Xs, ys, test_size=0.2, random_state=42)

    Xtr = torch.tensor(X_tr, dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)

    class LSTMReg(nn.Module):
        def __init__(self, n_in, hidden=64):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, num_layers=2, batch_first=True, dropout=0.2)
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    model = LSTMReg(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    t0 = time.time()
    model.train()
    for epoch in range(30):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    train_time = time.time() - t0

    # calibration residuals (|true - pred| on held-out calibration set)
    model.eval()
    with torch.no_grad():
        cal_pred = model(torch.tensor(X_cal, dtype=torch.float32).to(device)).cpu().numpy().ravel()
    calib_resid = np.abs(y_cal - cal_pred)

    # test predictions (one per engine, at its last cycle)
    test_seqs = last_sequences(test_X, test_engine, test_cycle, SEQ_LEN, n_features)
    with torch.no_grad():
        lstm_pred = model(torch.tensor(test_seqs, dtype=torch.float32).to(device)).cpu().numpy().ravel()
    lstm_pred = np.clip(lstm_pred, 0, RUL_CLIP)

    results["LSTM"] = (*metrics(y_test, lstm_pred), train_time)
    return lstm_pred, calib_resid


if __name__ == "__main__":
    main()
