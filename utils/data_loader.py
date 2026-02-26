# utils/data_loader.py
"""
Unified data loading utilities for centralized & federated training.
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ── SECOM Preprocessing ────────────────────────────────────

def load_secom_raw(data_dir: str = "data"):
    """Load raw SECOM feature + label files."""
    X = pd.read_csv(
        os.path.join(data_dir, "secom.data"),
        sep=" ", header=None, engine="python",
    )
    y = pd.read_csv(
        os.path.join(data_dir, "secom_labels.data"),
        sep=" ", header=None, engine="python",
    )
    y = y.iloc[:, 0]
    return X, y


def preprocess_secom(X, y):
    """Clean, impute, scale, and convert labels."""
    X = X.replace("NaN", np.nan).astype(float)

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_df = pd.DataFrame(X_scaled, columns=[f"f{i}" for i in range(X.shape[1])])
    y_clean = y.replace({-1: 0})

    X_df["label"] = y_clean.values
    return X_df


# ── Windowing ──────────────────────────────────────────────

def create_windows(df: pd.DataFrame, seq_len: int = 10):
    """Create sliding windows of length seq_len from the dataset."""
    X = df.drop("label", axis=1).values
    y = df["label"].values

    Xw, yw = [], []
    for i in range(len(X) - seq_len):
        Xw.append(X[i : i + seq_len])
        yw.append(y[i + seq_len - 1])

    return np.array(Xw, dtype=np.float32), np.array(yw, dtype=np.float32)


# ── Client Data Split ─────────────────────────────────────

def split_into_clients(Xw, yw, n_clients=3, test_size=0.2, seed=42):
    """Split windowed data into n_clients IID partitions."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(Xw))
    shards = np.array_split(indices, n_clients)

    client_data = {}
    for i, shard in enumerate(shards, start=1):
        X_train, X_test, y_train, y_test = train_test_split(
            Xw[shard], yw[shard], test_size=test_size, random_state=seed,
        )
        client_data[i] = {
            "X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test,
        }
    return client_data


def save_client_data(client_data: dict, client_dir: str):
    """Save client datasets to .npy files."""
    os.makedirs(client_dir, exist_ok=True)
    for cid, data in client_data.items():
        for key, arr in data.items():
            np.save(os.path.join(client_dir, f"client{cid}_{key}.npy"), arr)


# ── Load Pre-saved Client Data ────────────────────────────

def load_client_data(client_dir: str, cid: int):
    """Load a single client's train/test data from disk."""
    return {
        "X_train": np.load(os.path.join(client_dir, f"client{cid}_X_train.npy")),
        "y_train": np.load(os.path.join(client_dir, f"client{cid}_y_train.npy")),
        "X_test":  np.load(os.path.join(client_dir, f"client{cid}_X_test.npy")),
        "y_test":  np.load(os.path.join(client_dir, f"client{cid}_y_test.npy")),
    }


def load_all_clients_train(client_dir: str):
    """Concatenate all client training data (for centralized baseline)."""
    Xs, ys = [], []
    for path in sorted(glob.glob(os.path.join(client_dir, "*_X_train.npy"))):
        Xs.append(np.load(path))
        ys.append(np.load(path.replace("_X_train.npy", "_y_train.npy")))
    return np.concatenate(Xs), np.concatenate(ys)


def get_n_features(client_dir: str) -> int:
    """Infer number of features from the first client file found."""
    first = sorted(glob.glob(os.path.join(client_dir, "*_X_train.npy")))[0]
    return np.load(first).shape[2]
