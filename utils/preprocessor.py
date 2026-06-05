# utils/preprocessor.py
# CyberShield AI - Data Preprocessor
# Loads, cleans, and prepares CICIDS2017 dataset for ML training

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import pickle

# ── Path to dataset folder ──────────────────────────────────────────────────
DATA_PATH = r"D:\PROJECTS\CyberThreatDetection\data\MachineLearningCVE"

# ── CSV files to load ────────────────────────────────────────────────────────
CSV_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

def load_data():
    """Load and combine all CSV files into one DataFrame."""
    print("[*] Loading dataset files...")
    dfs = []
    for f in CSV_FILES:
        path = os.path.join(DATA_PATH, f)
        print(f"    Loading: {f}")
        df = pd.read_csv(path, low_memory=False)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"[+] Total records loaded: {len(combined):,}")
    return combined

def clean_data(df):
    """Clean the dataset - remove nulls, infinities, duplicates."""
    print("[*] Cleaning data...")
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with NaN
    df.dropna(inplace=True)
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    print(f"[+] Records after cleaning: {len(df):,}")
    return df

def encode_labels(df):
    """Encode attack labels into numeric classes."""
    print("[*] Encoding labels...")
    label_col = " Label" if " Label" in df.columns else "Label"
    
    # Simplify labels into main attack categories
    label_map = {
        "BENIGN": "BENIGN",
        "DDoS": "DDoS",
        "PortScan": "PortScan",
        "Bot": "Bot",
        "Web Attack \x96 Brute Force": "WebAttack",
        "Web Attack \x96 XSS": "WebAttack",
        "Web Attack \x96 Sql Injection": "WebAttack",
        "Infiltration": "Infiltration",
        "FTP-Patator": "BruteForce",
        "SSH-Patator": "BruteForce",
        "DoS slowloris": "DoS",
        "DoS Slowhttptest": "DoS",
        "DoS Hulk": "DoS",
        "DoS GoldenEye": "DoS",
        "Heartbleed": "Heartbleed",
    }
    df[label_col] = df[label_col].str.strip().map(label_map).fillna("Other")
    
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df[label_col])
    
    print(f"[+] Attack categories: {list(le.classes_)}")
    
    # Save label encoder for later use
    with open("models/saved_model/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print("[+] Label encoder saved.")
    return df, le

def prepare_features(df):
    """Select features and split into train/test sets."""
    print("[*] Preparing features...")
    label_col = " Label" if " Label" in df.columns else "Label"
    
    # Drop non-numeric and label columns
    drop_cols = [label_col, "label_encoded"]
    X = df.drop(columns=drop_cols, errors="ignore")
    
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    y = df["label_encoded"]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    with open("models/saved_model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("[+] Scaler saved.")
    
    # Save feature names
    with open("models/saved_model/feature_names.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)
    print(f"[+] Features used: {len(X.columns)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[+] Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, list(X.columns)

def run_preprocessing():
    """Full preprocessing pipeline."""
    os.makedirs("models/saved_model", exist_ok=True)
    df = load_data()
    df = clean_data(df)
    df, le = encode_labels(df)
    X_train, X_test, y_train, y_test, features = prepare_features(df)
    print("\n[✓] Preprocessing complete!")
    return X_train, X_test, y_train, y_test, features, le

if __name__ == "__main__":
    run_preprocessing()