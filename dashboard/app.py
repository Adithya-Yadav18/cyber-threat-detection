# dashboard/app.py
# Live Streamlit Dashboard for CyberShield AI
# Junior Developer: Sarvani

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="CyberShield AI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS STYLING
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00ff88;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING FUNCTIONS
# ─────────────────────────────────────────────

def load_from_sqlite(limit=100):
    """Load recent threats from SQLite database."""
    try:
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "database", "threats.db"
        )
        if not os.path.exists(db_path):
            return pd.DataFrame()
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            f"SELECT * FROM threats ORDER BY id DESC LIMIT {limit}",
            conn
        )
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()


def load_from_mongodb(limit=100):
    """Load recent threats from MongoDB."""
    try:
        from pymongo import MongoClient
        from dotenv import load_dotenv
        load_dotenv()
        client = MongoClient(
            os.getenv("MONGO_URI", "mongodb://localhost:27017/"),
            serverSelectionTimeoutMS=2000
        )
        db = client["cybershield"]
        threats = list(
            db["threats"].find({}, {"_id": 0})
            .sort("timestamp", -1)
            .limit(limit)
        )
        return pd.DataFrame(threats) if threats else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def get_threat_data(limit=100):
    """Try MongoDB first, fall back to SQLite."""
    df = load_from_mongodb(limit)
    if df.empty:
        df = load_from_sqlite(limit)
    return df


def get_simulated_data():
    """Generate simulated threat data matching Yujiro's exact data structure."""
    import random

    # Exact attack types from Yujiro's label encoder
    attack_types = [
        "DDoS", "DoS", "BruteForce", "PortScan",
        "WebAttack", "Infiltration", "Bot", "BENIGN"
    ]
    # Exact severity values from Yujiro's threat_classifier.py
    severity_map = {
        "DDoS":        "CRITICAL",
        "DoS":         "CRITICAL",
        "Infiltration":"CRITICAL",
        "Bot":         "HIGH",
        "BruteForce":  "HIGH",
        "WebAttack":   "HIGH",
        "PortScan":    "MEDIUM",
        "BENIGN":      "SAFE"
    }
    protocols = ["TCP", "UDP", "ICMP", "HTTP"]

    data = []
    for _ in range(50):
        attack = random.choice(attack_types)
        confidence = round(random.uniform(75.0, 99.0), 2)
        data.append({
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attack_type": attack,
            "confidence":  confidence,
            "severity":    severity_map[attack],
            "is_threat":   0 if attack == "BENIGN" else 1,
            "source_ip":   f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
            "dest_ip":     f"10.0.{random.randint(1,10)}.{random.randint(1,50)}",
            "protocol":    random.choice(protocols),
            "packet_rate": round(random.uniform(100, 10000), 2),
            "byte_rate":   round(random.uniform(1000, 100000), 2)
        })
    return pd.DataFrame(data)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shield.png", width=80)
    st.title("🛡️ CyberShield AI")
    st.markdown("---")

    # Mode toggle
    mode = st.radio(
        "📡 Dashboard Mode",
        ["🔴 Live Monitor", "🟡 Simulation Mode"],
        index=1
    )

    st.markdown("---")

    # Auto refresh toggle
    auto_refresh = st.checkbox("🔄 Auto Refresh (3s)", value=False)

    # Severity filter — matches Yujiro's exact severity values
    st.markdown("### 🔍 Filters")
    severity_filter = st.multiselect(
        "Filter by Severity",
        ["CRITICAL", "HIGH", "MEDIUM", "SAFE"],
        default=["CRITICAL", "HIGH", "MEDIUM", "SAFE"]
    )

    # Threat limit
    threat_limit = st.slider("Max Threats to Show", 10, 200, 50)

    st.markdown("---")
    st.markdown("**👩‍💻 Junior Dev:** Sarvani")
    st.markdown("**👨‍💻 Senior Dev:** Yujiro")
    st.markdown("**📅 CIS Internship 2024**")


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

if "Simulation" in mode:
    df = get_simulated_data()
    st.info("🟡 Running in Simulation Mode — showing demo data")
else:
    df = get_threat_data(threat_limit)
    if df.empty:
        st.warning("⚠️ No live data found — switching to simulation mode")
        df = get_simulated_data()
    else:
        st.success("🔴 Connected to live threat database")

# Apply severity filter
if not df.empty and "severity" in df.columns:
    df = df[df["severity"].isin(severity_filter)]


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("🛡️ CyberShield AI — Threat Detection Dashboard")
st.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown("---")


# ─────────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)

total_events   = len(df)
total_threats  = len(df[df["is_threat"] == 1]) if "is_threat" in df.columns else 0
critical_count = len(df[df["severity"] == "CRITICAL"]) if "severity" in df.columns else 0
high_count     = len(df[df["severity"] == "HIGH"]) if "severity" in df.columns else 0
avg_confidence = df["confidence"].mean() if "confidence" in df.columns else 0

with col1:
    st.metric("📊 Total Events", total_events)
with col2:
    st.metric("🚨 Threats Detected", total_threats)
with col3:
    st.metric("🔴 Critical", critical_count)
with col4:
    st.metric("🟠 High", high_count)
with col5:
    # Confidence is stored as percentage (e.g. 97.5) in Yujiro's system
    st.metric("🎯 Avg Confidence", f"{avg_confidence:.1f}%")

st.markdown("---")


# ─────────────────────────────────────────────
# CHARTS ROW
# ─────────────────────────────────────────────

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🥧 Attack Type Breakdown")
    if not df.empty and "attack_type" in df.columns:
        attack_counts = df["attack_type"].value_counts().reset_index()
        attack_counts.columns = ["Attack Type", "Count"]
        fig_pie = px.pie(
            attack_counts,
            values="Count",
            names="Attack Type",
            color_discrete_sequence=px.colors.sequential.RdBu,
            hole=0.4
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No data available")

with col_right:
    st.subheader("📊 Severity Distribution")
    if not df.empty and "severity" in df.columns:
        severity_counts = df["severity"].value_counts().reset_index()
        severity_counts.columns = ["Severity", "Count"]
        severity_colors = {
            "CRITICAL": "#ff0000",
            "HIGH":     "#ff6600",
            "MEDIUM":   "#ffaa00",
            "SAFE":     "#00aa00"
        }
        fig_bar = px.bar(
            severity_counts,
            x="Severity",
            y="Count",
            color="Severity",
            color_discrete_map=severity_colors,
            category_orders={"Severity": ["CRITICAL", "HIGH", "MEDIUM", "SAFE"]}
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No data available")

st.markdown("---")


# ─────────────────────────────────────────────
# LIVE THREAT FEED
# ─────────────────────────────────────────────

st.subheader("🔴 Live Threat Feed")

if not df.empty:
    threat_df = df[df["is_threat"] == 1] if "is_threat" in df.columns else df

    display_cols = [c for c in
        ["timestamp", "attack_type", "severity", "confidence",
         "source_ip", "dest_ip", "protocol", "packet_rate"]
        if c in threat_df.columns]

    if display_cols:
        st.dataframe(
            threat_df[display_cols].head(20),
            use_container_width=True,
            hide_index=True
        )
else:
    st.info("No threats detected yet")

st.markdown("---")


# ─────────────────────────────────────────────
# MODEL VOTES SECTION
# ─────────────────────────────────────────────

st.subheader("🤖 Ensemble Model Votes (Sample)")

vote_col1, vote_col2, vote_col3 = st.columns(3)

with vote_col1:
    st.markdown("**🧠 TensorFlow DNN**")
    st.metric("Prediction", "DDoS")
    st.metric("Confidence", "96.80%")
    st.metric("Weight", "35%")

with vote_col2:
    st.markdown("**🌲 XGBoost**")
    st.metric("Prediction", "DDoS")
    st.metric("Confidence", "99.20%")
    st.metric("Weight", "40%")

with vote_col3:
    st.markdown("**🔄 LSTM**")
    st.metric("Prediction", "DDoS")
    st.metric("Confidence", "94.50%")
    st.metric("Weight", "25%")

st.markdown("---")


# ─────────────────────────────────────────────
# SHAP EXPLANATION SECTION
# ─────────────────────────────────────────────

st.subheader("🔍 SHAP Explainability — Why Was This Flagged?")

shap_col1, shap_col2 = st.columns([1, 2])

with shap_col1:
    st.markdown("""
    **What is SHAP?**

    SHAP explains *why* the AI flagged a network packet as an attack.

    Each feature gets a score:
    - 🔴 **Positive** → pushes toward threat
    - 🟢 **Negative** → pushes away from threat

    The bigger the bar, the more important the feature.
    """)

with shap_col2:
    # SHAP waterfall chart using Yujiro's real feature names
    sample_features = [
        "Flow Packets/s",
        "Flow Bytes/s",
        "Fwd Packet Length Max",
        "Bwd Packet Length Max",
        "Flow Duration",
        "Fwd Packets/s",
        "Bwd Packets/s"
    ]
    sample_shap_values = [0.45, 0.32, 0.28, -0.15, 0.12, -0.08, 0.05]

    fig_shap = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="h",
        measure=["relative"] * len(sample_features),
        y=sample_features,
        x=sample_shap_values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#ff4444"}},
        decreasing={"marker": {"color": "#44ff88"}}
    ))
    fig_shap.update_layout(
        title="Feature Contributions (Sample DDoS Attack)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=300
    )
    st.plotly_chart(fig_shap, use_container_width=True)

st.markdown("---")


# ─────────────────────────────────────────────
# AI INCIDENT SUMMARY SECTION
# ─────────────────────────────────────────────

st.subheader("🤖 AI Incident Summary (Groq API)")

ai_col1, ai_col2 = st.columns(2)

with ai_col1:
    st.markdown("**Sample Critical Incident:**")
    st.error("""
    🚨 **CRITICAL — DDoS Attack Detected**

    A high-volume DDoS attack was detected from IP 192.168.47.231
    at 14:32:05 with 97.5% confidence. The ensemble of DNN, XGBoost,
    and LSTM models all agreed on the classification. Immediate IP
    blocking and DDoS mitigation activation is strongly recommended.
    """)

with ai_col2:
    st.markdown("**Sample High Incident:**")
    st.warning("""
    ⚠️ **HIGH — BruteForce Attack Detected**

    A brute force authentication attack was identified from IP 10.0.3.17
    at 14:35:22 with 89.3% confidence. Multiple failed login attempts
    were recorded targeting SSH port 22. Account lockout and IP blocking
    are recommended as immediate response actions.
    """)

st.markdown("---")


# ─────────────────────────────────────────────
# EMAIL ALERT LOG
# ─────────────────────────────────────────────

st.subheader("📧 Email Alert Log")

alert_data = {
    "Time":        ["14:32:05", "14:35:22", "14:41:10"],
    "Attack Type": ["DDoS", "BruteForce", "Infiltration"],
    "Severity":    ["CRITICAL", "HIGH", "CRITICAL"],
    "Recipient":   ["security@company.com"] * 3,
    "Status":      ["✅ Sent", "✅ Sent", "✅ Sent"]
}
st.dataframe(
    pd.DataFrame(alert_data),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")


# ─────────────────────────────────────────────
# AUTO REFRESH
# ─────────────────────────────────────────────

if auto_refresh:
    time.sleep(3)
    st.rerun()