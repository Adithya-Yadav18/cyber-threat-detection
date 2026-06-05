# main.py
# CyberShield AI - Main CrewAI Orchestrator
# Connects all 5 agents into one autonomous pipeline

import os
import sys
import time
import json
from datetime import datetime
from dotenv import load_dotenv
from crewai import Crew, Process

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.monitor_agent import (
    create_monitor_agent,
    create_monitor_task,
    generate_simulated_traffic
)
from agents.detection_agent import (
    create_detection_agent,
    create_detection_task,
    analyze_traffic,
    format_threat_report
)

# ── Database logging (MongoDB) ────────────────────────────────────────────────
def save_to_mongodb(result: dict):
    """Save threat result to MongoDB with SQLite fallback."""
    # Try MongoDB first
    try:
        from pymongo import MongoClient
        client = MongoClient(
            os.getenv("MONGO_URI"),
            serverSelectionTimeoutMS=3000,
            tlsAllowInvalidCertificates=True
        )
        db = client["cybershield"]
        collection = db["threats"]
        doc = {
            "timestamp":   result.get("timestamp", datetime.now().isoformat()),
            "attack_type": result.get("attack_type"),
            "confidence":  result.get("confidence"),
            "severity":    result.get("severity"),
            "is_threat":   result.get("is_threat"),
            "source_ip":   result.get("source_ip"),
            "dest_ip":     result.get("dest_ip"),
            "protocol":    result.get("protocol"),
            "packet_rate": result.get("packet_rate"),
            "byte_rate":   result.get("byte_rate"),
            "model_votes": str(result.get("model_votes")),
        }
        collection.insert_one(doc)
        print(f"[+] Saved to MongoDB: {result['attack_type']}")
        return
    except Exception:
        pass

    # Fallback to SQLite
    try:
        import sqlite3
        os.makedirs("database", exist_ok=True)
        conn = sqlite3.connect("database/threats.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                attack_type TEXT,
                confidence REAL,
                severity TEXT,
                is_threat INTEGER,
                source_ip TEXT,
                dest_ip TEXT,
                protocol TEXT,
                packet_rate REAL,
                byte_rate REAL,
                model_votes TEXT
            )
        """)
        cursor.execute("""
            INSERT INTO threats VALUES (
                NULL,?,?,?,?,?,?,?,?,?,?,?
            )
        """, (
            result.get("timestamp", datetime.now().isoformat()),
            result.get("attack_type"),
            result.get("confidence"),
            result.get("severity"),
            int(result.get("is_threat", False)),
            result.get("source_ip"),
            result.get("dest_ip"),
            result.get("protocol"),
            result.get("packet_rate"),
            result.get("byte_rate"),
            str(result.get("model_votes"))
        ))
        conn.commit()
        conn.close()
        print(f"[+] Saved to SQLite: {result['attack_type']}")
    except Exception as e:
        print(f"[!] Save failed: {e}")

# ── Single threat analysis pipeline ──────────────────────────────────────────
def run_threat_pipeline(attack_type=None, use_crewai=False):
    """
    Run one full threat detection cycle.
    attack_type: force a specific attack for testing
    use_crewai: use full CrewAI agent conversation (slower but shows agent thinking)
    """
    print("\n" + "="*60)
    print(f"  CyberShield AI - Threat Detection Cycle")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # ── Step 1: Monitor Agent captures traffic ────────────────────────────────
    print("\n[AGENT 1] Traffic Monitor Agent — Capturing traffic...")
    traffic_sample = generate_simulated_traffic(attack_type)
    print(f"  Source IP : {traffic_sample['_source_ip']}")
    print(f"  Protocol  : {traffic_sample['_protocol']}")
    print(f"  Simulated : {traffic_sample['_simulated_type']}")

    # ── Step 2: Detection Agent classifies threat ─────────────────────────────
    print("\n[AGENT 2] Threat Detection Agent — Running ML ensemble...")
    result = analyze_traffic(traffic_sample)
    print(format_threat_report(result))

    # ── Step 3: Save to MongoDB ───────────────────────────────────────────────
    save_to_mongodb(result)

    # ── Step 4: CrewAI conversation (optional, for demo) ─────────────────────
    if use_crewai and result["is_threat"]:
        print("\n[CREWAI] Activating agent collaboration...")
        try:
            monitor_agent   = create_monitor_agent()
            detection_agent = create_detection_agent()

            monitor_task   = create_monitor_task(monitor_agent, traffic_sample)
            detection_task = create_detection_task(detection_agent, result)

            crew = Crew(
                agents=[monitor_agent, detection_agent],
                tasks=[monitor_task, detection_task],
                process=Process.sequential,
                verbose=True
            )
            crew_output = crew.kickoff()
            print("\n[CREWAI OUTPUT]")
            print(crew_output)
        except Exception as e:
            print(f"[!] CrewAI conversation error: {e}")

    return result

# ── Continuous monitoring loop ────────────────────────────────────────────────
def run_continuous_monitor(cycles=10, interval=2, use_crewai=False):
    """
    Run continuous threat monitoring.
    cycles: number of traffic samples to analyze
    interval: seconds between each cycle
    """
    print("\n" + "🛡️ "*20)
    print("  CYBERSHIELD AI — CONTINUOUS THREAT MONITOR STARTED")
    print("🛡️ "*20)
    print(f"  Monitoring {cycles} cycles | {interval}s interval")
    print(f"  Press Ctrl+C to stop\n")

    results = []
    threats_detected = 0

    for i in range(cycles):
        print(f"\n[Cycle {i+1}/{cycles}]")
        try:
            result = run_threat_pipeline(use_crewai=use_crewai)
            results.append(result)
            if result["is_threat"]:
                threats_detected += 1
                print(f"  ⚠️  THREAT #{threats_detected}: {result['attack_type']} "
                      f"({result['severity']}) — {result['confidence']}% confidence")
            else:
                print(f"  ✅ Traffic normal")
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[!] Monitoring stopped by user.")
            break
        except Exception as e:
            print(f"[!] Cycle error: {e}")
            continue

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  MONITORING SESSION SUMMARY")
    print("="*60)
    print(f"  Total cycles     : {len(results)}")
    print(f"  Threats detected : {threats_detected}")
    print(f"  Normal traffic   : {len(results) - threats_detected}")
    if results:
        threat_types = {}
        for r in results:
            t = r["attack_type"]
            threat_types[t] = threat_types.get(t, 0) + 1
        print(f"\n  Breakdown:")
        for t, count in sorted(threat_types.items(),
                                key=lambda x: x[1], reverse=True):
            print(f"    {t:15} : {count}")
    print("="*60)
    return results

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CyberShield AI")
    parser.add_argument("--cycles",   type=int,  default=5,
                        help="Number of monitoring cycles")
    parser.add_argument("--interval", type=float, default=1,
                        help="Seconds between cycles")
    parser.add_argument("--crewai",   action="store_true",
                        help="Enable full CrewAI agent conversations")
    parser.add_argument("--attack",   type=str,  default=None,
                        help="Force specific attack type for testing")
    args = parser.parse_args()

    run_continuous_monitor(
        cycles=args.cycles,
        interval=args.interval,
        use_crewai=args.crewai
    )