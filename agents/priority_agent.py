# agents/priority_agent.py
# Scores the severity of detected threats: Low / Medium / High / Critical
# Junior Developer: Sarvani

from datetime import datetime

# ─────────────────────────────────────────────
# SEVERITY RULES
# ─────────────────────────────────────────────

# Attack types and their base severity scores (0-100)
ATTACK_SEVERITY_SCORES = {
    "DDoS": 90,
    "PortScan": 50,
    "BruteForce": 70,
    "DoS": 85,
    "WebAttack": 75,
    "Infiltration": 95,
    "Botnet": 88,
    "Benign": 0
}

# Confidence thresholds that modify severity
CONFIDENCE_BOOST = {
    "high": 0.90,    # confidence >= 90% → boost severity
    "medium": 0.70,  # confidence >= 70% → no change
    "low": 0.50      # confidence < 70% → reduce severity
}

# Final severity level thresholds
SEVERITY_LEVELS = {
    "Critical": 85,
    "High": 65,
    "Medium": 40,
    "Low": 0
}

# Recommended actions per severity level
RECOMMENDED_ACTIONS = {
    "Critical": "Block source IP immediately, isolate affected systems, notify security team",
    "High": "Investigate source IP, enable enhanced monitoring, prepare incident report",
    "Medium": "Monitor traffic closely, log all activity, review firewall rules",
    "Low": "Log the event, continue standard monitoring"
}


# ─────────────────────────────────────────────
# PRIORITY SCORING FUNCTION
# ─────────────────────────────────────────────

def calculate_priority(attack_type: str, confidence: float) -> dict:
    """
    Calculates the severity priority of a detected threat.

    Args:
        attack_type (str): Type of attack detected (e.g. "DDoS", "BruteForce")
        confidence (float): Model confidence score between 0 and 1

    Returns:
        dict: Priority result containing severity, score, and recommended action
        Example:
        {
            "severity": "Critical",
            "priority_score": 92,
            "recommended_action": "Block source IP immediately...",
            "timestamp": "2024-01-01 12:00:00"
        }
    """

    # ── Get base score for attack type ──
    base_score = ATTACK_SEVERITY_SCORES.get(attack_type, 50)

    # ── Adjust score based on confidence ──
    if confidence >= CONFIDENCE_BOOST["high"]:
        # High confidence — boost score by 5 points
        adjusted_score = min(base_score + 5, 100)
    elif confidence >= CONFIDENCE_BOOST["medium"]:
        # Medium confidence — keep score as is
        adjusted_score = base_score
    else:
        # Low confidence — reduce score by 10 points
        adjusted_score = max(base_score - 10, 0)

    # ── Determine severity level ──
    if adjusted_score >= SEVERITY_LEVELS["Critical"]:
        severity = "Critical"
    elif adjusted_score >= SEVERITY_LEVELS["High"]:
        severity = "High"
    elif adjusted_score >= SEVERITY_LEVELS["Medium"]:
        severity = "Medium"
    else:
        severity = "Low"

    # ── Get recommended action ──
    action = RECOMMENDED_ACTIONS.get(severity, "Monitor and log")

    # ── Build result ──
    result = {
        "severity": severity,
        "priority_score": adjusted_score,
        "recommended_action": action,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    print(f"[PRIORITY] {attack_type} → {severity} (score: {adjusted_score}, confidence: {confidence:.2%})")
    return result


# ─────────────────────────────────────────────
# BATCH PRIORITY SCORING
# ─────────────────────────────────────────────

def prioritize_threats(threats: list) -> list:
    """
    Scores severity for a list of threats and sorts by priority.

    Args:
        threats (list): List of threat dictionaries

    Returns:
        list: Threats sorted by priority score (highest first)
    """
    prioritized = []

    for threat in threats:
        attack_type = threat.get("attack_type", "Unknown")
        confidence = threat.get("confidence", 0.5)

        # Calculate priority
        priority = calculate_priority(attack_type, confidence)

        # Merge priority info into threat data
        threat.update(priority)
        prioritized.append(threat)

    # Sort by priority score highest first
    prioritized.sort(key=lambda x: x.get("priority_score", 0), reverse=True)

    print(f"[PRIORITY] Scored and sorted {len(prioritized)} threats")
    return prioritized