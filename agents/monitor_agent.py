# agents/monitor_agent.py
# CyberShield AI - Traffic Monitor Agent
# Agent 1: Monitors network traffic and extracts features

from crewai import Agent, Task
import pandas as pd
import numpy as np
import pickle
import os
import random
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_PATH = os.path.join(BASE_DIR, "models", "saved_model", "feature_names.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "MachineLearningCVE")

# ── CSV files mapped to attack types ─────────────────────────────────────────
CSV_MAP = {
    "DDoS":       "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "PortScan":   "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "BENIGN":     "Monday-WorkingHours.pcap_ISCX.csv",
    "BruteForce": "Tuesday-WorkingHours.pcap_ISCX.csv",
    "DoS":        "Wednesday-workingHours.pcap_ISCX.csv",
    "WebAttack":  "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
}

# ── Cache for loaded samples ──────────────────────────────────────────────────
_sample_cache = {}

def get_feature_names():
    with open(FEATURE_PATH, "rb") as f:
        return pickle.load(f)

def load_samples_for_attack(attack_type: str, n: int = 500):
    """Load real samples from dataset for given attack type."""
    if attack_type in _sample_cache:
        return _sample_cache[attack_type]

    csv_file = CSV_MAP.get(attack_type, CSV_MAP["BENIGN"])
    path = os.path.join(DATA_PATH, csv_file)

    print(f"[*] Loading real samples for {attack_type}...")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    label_col = " Label" if " Label" in df.columns else "Label"

    # Filter by attack type
    label_map = {
        "BENIGN": "BENIGN",
        "DDoS": "DDoS",
        "PortScan": "PortScan",
        "BruteForce": ["FTP-Patator", "SSH-Patator"],
        "DoS": ["DoS slowloris", "DoS Slowhttptest", "DoS Hulk", "DoS GoldenEye"],
        "WebAttack": ["Web Attack \x96 Brute Force", "Web Attack \x96 XSS",
                      "Web Attack \x96 Sql Injection"],
    }

    target = label_map.get(attack_type, attack_type)
    if isinstance(target, list):
        mask = df[label_col].str.strip().isin(target)
    else:
        mask = df[label_col].str.strip() == target

    filtered = df[mask]
    if len(filtered) == 0:
        filtered = df

    # Sample n rows
    samples = filtered.sample(min(n, len(filtered)), random_state=42)
    feature_names = get_feature_names()
    available = [f for f in feature_names if f in samples.columns]
    samples = samples[available].values.tolist()

    _sample_cache[attack_type] = samples
    print(f"[+] Loaded {len(samples)} real samples for {attack_type}")
    return samples

def generate_simulated_traffic(attack_type=None):
    """
    Generate a realistic traffic sample using real dataset values.
    """
    feature_names = get_feature_names()

    # Weighted random attack selection
    if attack_type is None:
        weights = [0.60, 0.12, 0.08, 0.08, 0.06, 0.04, 0.01, 0.01]
        attack_type = random.choices(
            ["BENIGN", "DDoS", "PortScan", "DoS",
             "BruteForce", "WebAttack", "Bot", "Other"],
            weights=weights
        )[0]

    # Use real data for known types, fallback for unknown
    known_types = list(CSV_MAP.keys())
    if attack_type in known_types:
        try:
            samples = load_samples_for_attack(attack_type)
            raw = random.choice(samples)
            available_features = [
                f for f in feature_names
                if f in get_feature_names()
            ]
            # Build sample dict
            feature_names_list = get_feature_names()
            available = [f for f in feature_names_list if f in feature_names_list]
            sample = {}
            for i, fname in enumerate(feature_names_list):
                if i < len(raw):
                    sample[fname] = float(raw[i])
                else:
                    sample[fname] = 0.0
        except Exception as e:
            print(f"[!] Fallback to random for {attack_type}: {e}")
            sample = {fname: random.uniform(0, 100) for fname in feature_names}
    else:
        sample = {fname: random.uniform(0, 100) for fname in feature_names}

    # Add metadata
    sample["_simulated_type"] = attack_type
    sample["_timestamp"] = datetime.now().isoformat()
    sample["_source_ip"] = (
        f"{random.randint(1,255)}.{random.randint(0,255)}."
        f"{random.randint(0,255)}.{random.randint(1,254)}"
    )
    sample["_dest_ip"] = (
        f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
    )
    sample["_protocol"] = random.choice(["TCP", "UDP", "ICMP"])
    return sample

def get_traffic_summary(sample: dict) -> str:
    """Generate a human-readable traffic summary."""
    return (
        f"Traffic Sample Captured:\n"
        f"  Source IP    : {sample.get('_source_ip', 'Unknown')}\n"
        f"  Dest IP      : {sample.get('_dest_ip', 'Unknown')}\n"
        f"  Protocol     : {sample.get('_protocol', 'Unknown')}\n"
        f"  Timestamp    : {sample.get('_timestamp', 'Unknown')}\n"
        f"  Packet Rate  : {sample.get('Flow Packets/s', 0):.1f} pkt/s\n"
        f"  Byte Rate    : {sample.get('Flow Bytes/s', 0):.1f} bytes/s\n"
        f"  Fwd Packets  : {sample.get('Total Fwd Packets', 0):.0f}\n"
        f"  Back Packets : {sample.get('Total Backward Packets', 0):.0f}\n"
    )

# ── CrewAI Agent Definition ───────────────────────────────────────────────────
def create_monitor_agent():
    return Agent(
        role="Cybersecurity Traffic Monitor",
        goal=(
            "Continuously monitor network traffic, capture packet features, "
            "and provide structured traffic summaries for threat analysis."
        ),
        backstory=(
            "You are an elite network traffic analyst with 15 years of experience "
            "in enterprise security operations centers. You have monitored millions "
            "of network flows and can instantly identify suspicious traffic patterns."
        ),
        verbose=True,
        allow_delegation=False,
    )

def create_monitor_task(agent, traffic_sample: dict):
    summary = get_traffic_summary(traffic_sample)
    return Task(
        description=(
            f"Analyze this network traffic sample and provide a structured summary:\n\n"
            f"{summary}\n"
            f"Identify any immediately obvious suspicious patterns."
        ),
        agent=agent,
        expected_output=(
            "A structured traffic summary with initial suspicion assessment."
        )
    )

# ── Standalone Test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Traffic Monitor Agent with real data...")
    print("="*50)
    for attack in ["BENIGN", "DDoS", "PortScan"]:
        sample = generate_simulated_traffic(attack)
        print(f"\nSimulated {attack} traffic:")
        print(get_traffic_summary(sample))
    print("[✓] Monitor Agent ready.")