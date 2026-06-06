# agents/response_agent.py
# Coordinates threat response: recommendations + Groq summary + alerts
# Junior Developer: Sarvani

from datetime import datetime
from utils.logger import log_threat
from utils.emailer import send_threat_alert
from utils.groq_summary import generate_incident_summary
from agents.priority_agent import calculate_priority

# ─────────────────────────────────────────────
# RESPONSE RULES
# ─────────────────────────────────────────────

# Detailed response playbook per attack type
RESPONSE_PLAYBOOK = {
    "DDoS": [
        "Enable rate limiting on all network interfaces",
        "Block source IP at firewall level",
        "Activate DDoS mitigation service",
        "Notify ISP about the attack",
        "Monitor bandwidth usage continuously"
    ],
    "BruteForce": [
        "Lock affected user accounts immediately",
        "Block source IP address",
        "Enable multi-factor authentication",
        "Review authentication logs",
        "Reset passwords for targeted accounts"
    ],
    "PortScan": [
        "Block scanning IP address",
        "Review open ports and close unnecessary ones",
        "Check for follow-up intrusion attempts",
        "Update firewall rules",
        "Monitor for further reconnaissance activity"
    ],
    "DoS": [
        "Block source IP immediately",
        "Enable traffic filtering",
        "Activate backup systems if needed",
        "Document attack patterns",
        "Contact upstream provider if attack persists"
    ],
    "WebAttack": [
        "Enable Web Application Firewall (WAF)",
        "Block malicious IP addresses",
        "Review and patch web application vulnerabilities",
        "Check for data exfiltration",
        "Audit web server logs"
    ],
    "Infiltration": [
        "Isolate affected systems immediately",
        "Revoke compromised credentials",
        "Conduct full forensic investigation",
        "Notify incident response team",
        "Preserve all system logs as evidence"
    ],
    "Botnet": [
        "Isolate infected machines from network",
        "Run malware removal tools",
        "Block command and control server IPs",
        "Reset all credentials on affected systems",
        "Conduct full security audit"
    ],
    "Benign": [
        "No action required",
        "Continue standard monitoring"
    ]
}


# ─────────────────────────────────────────────
# MAIN RESPONSE FUNCTION
# ─────────────────────────────────────────────

def handle_threat(threat_data: dict, shap_explanation: dict = None) -> dict:
    """
    Handles a detected threat end-to-end:
    1. Calculates priority and severity
    2. Gets response recommendations
    3. Generates AI incident summary
    4. Sends email alert
    5. Logs to file and MongoDB

    Args:
        threat_data (dict): Raw threat data from detection agent
        Example:
        {
            "attack_type": "DDoS",
            "confidence": 0.98,
            "source_ip": "192.168.1.1",
            "features": {...}
        }
        shap_explanation (dict): SHAP explanation from explainability agent

    Returns:
        dict: Complete incident report
    """

    print(f"\n[RESPONSE] Handling threat: {threat_data.get('attack_type', 'Unknown')}")

    # ── Step 1: Calculate priority ──
    attack_type = threat_data.get("attack_type", "Unknown")
    confidence = threat_data.get("confidence", 0.5)
    priority = calculate_priority(attack_type, confidence)

    # ── Step 2: Get response playbook ──
    playbook = RESPONSE_PLAYBOOK.get(attack_type, ["Monitor and investigate"])

    # ── Step 3: Build complete threat record ──
    incident = {
        "attack_type": attack_type,
        "confidence": confidence,
        "source_ip": threat_data.get("source_ip", "Unknown"),
        "severity": priority["severity"],
        "priority_score": priority["priority_score"],
        "recommended_action": priority["recommended_action"],
        "response_steps": playbook,
        "shap_explanation": shap_explanation if shap_explanation else {},
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # ── Step 4: Generate AI summary ──
    print("[RESPONSE] Generating AI incident summary...")
    ai_summary = generate_incident_summary(incident)
    incident["ai_summary"] = ai_summary

    # ── Step 5: Send email alert for High and Critical threats ──
    if priority["severity"] in ["High", "Critical"]:
        print(f"[RESPONSE] Sending email alert for {priority['severity']} threat...")
        email_sent = send_threat_alert(incident, ai_summary)
        incident["email_alert_sent"] = email_sent
    else:
        incident["email_alert_sent"] = False
        print(f"[RESPONSE] Severity is {priority['severity']} — no email alert needed")

    # ── Step 6: Log to file and MongoDB ──
    print("[RESPONSE] Logging threat to file and database...")
    log_threat(incident)

    print(f"[RESPONSE] ✅ Incident handled: {attack_type} | {priority['severity']}")
    return incident


# ─────────────────────────────────────────────
# BATCH RESPONSE HANDLER
# ─────────────────────────────────────────────

def handle_multiple_threats(threats: list) -> list:
    """
    Handles a batch of detected threats.

    Args:
        threats (list): List of threat dictionaries

    Returns:
        list: List of complete incident reports
    """
    incidents = []
    print(f"\n[RESPONSE] Processing {len(threats)} threats...")

    for threat in threats:
        incident = handle_threat(threat)
        incidents.append(incident)

    print(f"[RESPONSE] ✅ All {len(incidents)} threats handled")
    return incidents