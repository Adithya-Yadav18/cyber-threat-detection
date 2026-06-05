# utils/logger.py
# Handles logging of detected threats to file and MongoDB
# Junior Developer: Sarvani

import logging
import os
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─────────────────────────────────────────────
# FILE LOGGER SETUP
# ─────────────────────────────────────────────

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure the file logger
logging.basicConfig(
    filename="logs/threats.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# MONGODB SETUP
# ─────────────────────────────────────────────

def get_mongo_collection():
    """Connect to MongoDB and return the threats collection."""
    try:
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        db = client["cyber_threat_db"]
        collection = db["threats"]
        return collection
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        return None


# ─────────────────────────────────────────────
# MAIN LOGGING FUNCTION
# ─────────────────────────────────────────────

def log_threat(threat_data: dict):
    """
    Logs a detected threat to both file and MongoDB.

    Args:
        threat_data (dict): Dictionary containing threat details
        Example:
        {
            "attack_type": "DDoS",
            "severity": "Critical",
            "confidence": 0.98,
            "source_ip": "192.168.1.1",
            "timestamp": "2024-01-01 12:00:00",
            "shap_explanation": {...},
            "recommended_action": "Block IP immediately"
        }
    """

    # Add timestamp if not present
    if "timestamp" not in threat_data:
        threat_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Log to file ──
    try:
        logger.warning(
            f"THREAT DETECTED | "
            f"Type: {threat_data.get('attack_type', 'Unknown')} | "
            f"Severity: {threat_data.get('severity', 'Unknown')} | "
            f"Confidence: {threat_data.get('confidence', 0):.2%} | "
            f"Source IP: {threat_data.get('source_ip', 'Unknown')} | "
            f"Action: {threat_data.get('recommended_action', 'None')}"
        )
        print(f"[LOG] Threat logged to file: {threat_data.get('attack_type')}")
    except Exception as e:
        print(f"[ERROR] File logging failed: {e}")

    # ── Log to MongoDB ──
    try:
        collection = get_mongo_collection()
        if collection is not None:
            collection.insert_one(threat_data)
            print(f"[LOG] Threat saved to MongoDB: {threat_data.get('attack_type')}")
        else:
            print("[WARNING] MongoDB unavailable — skipping database log")
    except Exception as e:
        print(f"[ERROR] MongoDB logging failed: {e}")


# ─────────────────────────────────────────────
# FETCH RECENT THREATS (for dashboard)
# ─────────────────────────────────────────────

def get_recent_threats(limit: int = 50):
    """
    Fetches the most recent threats from MongoDB.

    Args:
        limit (int): Number of recent threats to fetch (default 50)

    Returns:
        list: List of threat dictionaries
    """
    try:
        collection = get_mongo_collection()
        if collection is not None:
            threats = list(
                collection.find({}, {"_id": 0})
                .sort("timestamp", -1)
                .limit(limit)
            )
            return threats
        else:
            print("[WARNING] MongoDB unavailable — returning empty list")
            return []
    except Exception as e:
        print(f"[ERROR] Failed to fetch threats: {e}")
        return []