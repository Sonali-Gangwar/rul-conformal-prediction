"""
data_prep.py  —  Step 1 of the RUL project (FD001)

Loads the C-MAPSS FD001 train/test files, computes the RUL label, drops useless
sensors, scales features, and saves ready-to-use arrays for the baseline models.

Run:  python data_prep.py
Output: outputs/prepared.npz
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data"
OUT_DIR = "outputs"
RUL_CLIP = 125   # standard FD001 practice: cap RUL at 125 (Ayushi's slides mention this)

# Column names: engine id, cycle, 3 operating settings, 21 sensors
COLS = ["engine", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def load_raw(split):
    """Load train_FD001.txt or test_FD001.txt into a DataFrame."""
    path = os.path.join(DATA_DIR, f"{split}_FD001.txt")
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, :26]          # keep first 26 columns (some files have trailing blanks)
    df.columns = COLS
    return df


def add_train_rul(df):
    """For training engines (run to failure), RUL = max_cycle - current_cycle."""
    max_cycle = df.groupby("engine")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=RUL_CLIP)   # cap at 125
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train = add_train_rul(load_raw("train"))
    test = load_raw("test")
    true_rul = pd.read_csv(os.path.join(DATA_DIR, "RUL_FD001.txt"), header=None).iloc[:, 0].values

    # --- pick sensors that actually change (constant sensors carry no info) ---
    # In FD001 these sensors are flat and are dropped by convention:
    drop_sensors = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
    feature_cols = [c for c in COLS if c.startswith("s") and c not in drop_sensors]
    # also drop operating settings for FD001 (single condition -> not informative)

    # --- scale features using TRAIN statistics only (no leakage) ---
    scaler = StandardScaler().fit(train[feature_cols].values)
    train_X = scaler.transform(train[feature_cols].values)
    test_X = scaler.transform(test[feature_cols].values)

    # For the tabular models (RF/XGB) we use the per-row features directly.
    # For test, the label is only known at the LAST cycle of each engine.
    train_y = train["RUL"].values
    train_engine = train["engine"].values
    train_cycle = train["cycle"].values

    test_engine = test["engine"].values
    test_cycle = test["cycle"].values

    np.savez_compressed(
        os.path.join(OUT_DIR, "prepared.npz"),
        train_X=train_X, train_y=train_y,
        train_engine=train_engine, train_cycle=train_cycle,
        test_X=test_X, test_engine=test_engine, test_cycle=test_cycle,
        true_rul=true_rul, feature_cols=np.array(feature_cols),
    )
    print(f"Saved outputs/prepared.npz")
    print(f"  train rows: {train_X.shape[0]}, features: {train_X.shape[1]}")
    print(f"  test engines: {len(np.unique(test_engine))}, true_rul values: {len(true_rul)}")


if __name__ == "__main__":
    main()
