# utils/groq_summary.py
# Generates AI-powered incident summaries using Groq API
# Junior Developer: Sarvani

import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─────────────────────────────────────────────
# GROQ CLIENT SETUP
# ─────────────────────────────────────────────

def get_groq_client():
    """Initialize and return Groq client using API key from .env"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[WARNING] GROQ_API_KEY not found in .env file")
        return None
    return Groq(api_key=api_key)


# ─────────────────────────────────────────────
# MAIN SUMMARY FUNCTION
# ─────────────────────────────────────────────

def generate_incident_summary(threat_data: dict) -> str:
    """
    Generates an AI-powered incident summary for a detected threat.

    Args:
        threat_data (dict): Dictionary containing threat details
        Example:
        {
            "attack_type": "DDoS",
            "severity": "Critical",
            "confidence": 0.98,
            "source_ip": "192.168.1.1",
            "timestamp": "2024-01-01 12:00:00",
            "recommended_action": "Block IP immediately"
        }

    Returns:
        str: AI-generated incident summary
    """

    # ── Build the prompt ──
    prompt = f"""
You are a cybersecurity analyst. Write a short, professional incident summary report based on this threat detection:

- Attack Type: {threat_data.get('attack_type', 'Unknown')}
- Severity Level: {threat_data.get('severity', 'Unknown')}
- Confidence Score: {threat_data.get('confidence', 0):.2%}
- Source IP Address: {threat_data.get('source_ip', 'Unknown')}
- Detection Time: {threat_data.get('timestamp', 'Unknown')}
- Recommended Action: {threat_data.get('recommended_action', 'Under investigation')}

Write 3-4 sentences. Be concise, professional, and clear.
Start directly with the summary — no titles or headers needed.
"""

    # ── Call Groq API ──
    try:
        client = get_groq_client()
        if client is None:
            return _fallback_summary(threat_data)

        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional cybersecurity incident analyst. Write clear, concise incident summaries."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=200,
            temperature=0.3
        )

        summary = response.choices[0].message.content.strip()
        print(f"[GROQ] Summary generated for {threat_data.get('attack_type')} attack")
        return summary

    except Exception as e:
        print(f"[ERROR] Groq API call failed: {e}")
        return _fallback_summary(threat_data)


# ─────────────────────────────────────────────
# FALLBACK SUMMARY (if Groq API unavailable)
# ─────────────────────────────────────────────

def _fallback_summary(threat_data: dict) -> str:
    """
    Returns a basic summary if Groq API is unavailable.

    Args:
        threat_data (dict): Dictionary containing threat details

    Returns:
        str: Basic formatted summary
    """
    return (
        f"A {threat_data.get('severity', 'Unknown')} severity "
        f"{threat_data.get('attack_type', 'Unknown')} attack was detected "
        f"from IP {threat_data.get('source_ip', 'Unknown')} "
        f"at {threat_data.get('timestamp', 'Unknown')} "
        f"with {threat_data.get('confidence', 0):.2%} confidence. "
        f"Recommended action: {threat_data.get('recommended_action', 'Under investigation')}."
    )