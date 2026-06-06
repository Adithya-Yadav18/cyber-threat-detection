# agents/detection_agent.py
# CyberShield AI - Threat Detection Agent
# Agent 2: Runs ensemble ML models to classify threats

from crewai import Agent, Task
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.threat_classifier import ThreatClassifier

# ── Singleton classifier instance ────────────────────────────────────────────
_classifier = None

def get_classifier():
    """Load classifier once and reuse."""
    global _classifier
    if _classifier is None:
        _classifier = ThreatClassifier()
        _classifier.load_models()
    return _classifier

def analyze_traffic(sample: dict) -> dict:
    """
    Run full threat analysis on a traffic sample.
    Returns complete threat report.
    """
    classifier = get_classifier()

    # Remove metadata keys before prediction
    clean_sample = {
        k: v for k, v in sample.items()
        if not k.startswith("_")
    }

    # Run ensemble prediction
    result = classifier.predict(clean_sample)

    # Add metadata back to result
    result["source_ip"]   = sample.get("_source_ip", "Unknown")
    result["dest_ip"]     = sample.get("_dest_ip", "Unknown")
    result["protocol"]    = sample.get("_protocol", "Unknown")
    result["timestamp"]   = sample.get("_timestamp", "Unknown")
    result["packet_rate"] = round(sample.get("Flow Packets/s", 0), 2)
    result["byte_rate"]   = round(sample.get("Flow Bytes/s", 0), 2)

    return result

def format_threat_report(result: dict) -> str:
    """Format prediction result into readable threat report."""
    status = "🔴 THREAT DETECTED" if result["is_threat"] else "🟢 NORMAL TRAFFIC"

    severity_icons = {
        "CRITICAL": "🔴",
        "HIGH":     "🟠",
        "MEDIUM":   "🟡",
        "SAFE":     "🟢"
    }
    icon = severity_icons.get(result["severity"], "⚪")

    report = f"""
╔══════════════════════════════════════════════════╗
║           CYBERSHIELD AI - THREAT REPORT          ║
╚══════════════════════════════════════════════════╝
  Status       : {status}
  Attack Type  : {result['attack_type']}
  Confidence   : {result['confidence']}%
  Risk Level   : {icon} {result['severity']}
  Source IP    : {result.get('source_ip', 'N/A')}
  Dest IP      : {result.get('dest_ip', 'N/A')}
  Protocol     : {result.get('protocol', 'N/A')}
  Packet Rate  : {result.get('packet_rate', 0)} pkt/s
  Byte Rate    : {result.get('byte_rate', 0)} bytes/s
  Timestamp    : {result.get('timestamp', 'N/A')}

  Model Votes:
    DNN      → {result['model_votes']['dnn']['prediction']:12} ({result['model_votes']['dnn']['confidence']}%)
    XGBoost  → {result['model_votes']['xgboost']['prediction']:12} ({result['model_votes']['xgboost']['confidence']}%)
    LSTM     → {result['model_votes']['lstm']['prediction']:12} ({result['model_votes']['lstm']['confidence']}%)
══════════════════════════════════════════════════
"""
    return report

# ── CrewAI Agent Definition ───────────────────────────────────────────────────
def create_detection_agent():
    """Create and return the Threat Detection CrewAI Agent."""
    return Agent(
        role="AI Threat Detection Specialist",
        goal=(
            "Analyze network traffic features using ensemble ML models "
            "(TensorFlow DNN, XGBoost, LSTM) to accurately classify "
            "threats and determine attack types with confidence scores."
        ),
        backstory=(
            "You are an AI-powered threat detection engine trained on millions "
            "of real network attacks. You combine three state-of-the-art machine "
            "learning models to achieve near-perfect threat detection accuracy. "
            "Your ensemble approach ensures no attack goes undetected."
        ),
        verbose=True,
        allow_delegation=False,
    )

def create_detection_task(agent, result: dict):
    """Create a detection task for the agent."""
    report = format_threat_report(result)
    return Task(
        description=(
            f"Review this threat detection result and provide a professional "
            f"security assessment:\n{report}\n"
            f"Explain what this attack type means, how dangerous it is, "
            f"and what systems are typically targeted."
        ),
        agent=agent,
        expected_output=(
            "A professional security assessment explaining the detected threat, "
            "its danger level, typical targets, and attack methodology."
        )
    )

# ── Standalone Test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from agents.monitor_agent import generate_simulated_traffic

    print("Testing Threat Detection Agent...")
    print("="*50)

    # Test with different attack types
    for attack in ["BENIGN", "DDoS", "PortScan", "BruteForce"]:
        print(f"\nTesting: {attack}")
        sample = generate_simulated_traffic(attack)
        result = analyze_traffic(sample)
        print(format_threat_report(result))

    print("[✓] Detection Agent ready.")