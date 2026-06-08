# 🛡️ Enterprise Multi-Agent Cybersecurity Threat Detection & Response System

A real-time cybersecurity threat detection system built with ML models, autonomous AI agents, and a live monitoring dashboard.

## 👥 Team
- **Adithya** — Senior Developer (ML models, core agents, FastAPI backend, deployment)
- **Sarvani** — Junior Developer (supporting agents, dashboard, alerts, AI summaries)

## 🔍 What This System Does
- Monitors live network traffic using Scapy
- Detects cyber attacks using 3 ML models (TensorFlow DNN + XGBoost + LSTM)
- Uses 5 autonomous CrewAI agents that communicate with each other
- Displays everything on a live Streamlit dashboard
- Sends email alerts with AI-generated incident summaries
- Logs all threats to MongoDB in real time
- Explains AI decisions using SHAP (Explainable AI)
- Generates incident summaries using Groq API

## 🤖 The 5 Agents
| Agent | Owner | Job |
|-------|-------|-----|
| Traffic Monitor Agent | Yujiro | Captures live network packets using Scapy |
| Threat Detection Agent | Yujiro | Runs TensorFlow + XGBoost + LSTM models |
| Alert Priority Agent | Sarvani | Scores severity: Low / Medium / High / Critical |
| Explainability Agent | Sarvani | Uses SHAP to explain why a threat was flagged |
| Incident Response Agent | Sarvani | Recommends actions + triggers Groq AI summary |

## 🧰 Tech Stack
| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| ML Models | TensorFlow, XGBoost, LSTM |
| Agents | CrewAI |
| Dashboard | Streamlit |
| Backend | FastAPI |
| Database | MongoDB |
| Explainability | SHAP |
| AI Summaries | Groq API |
| Deployment | Docker + Render |

## 📁 Project Structure
cyber-threat-detection/
├── agents/
│   ├── monitor_agent.py        ← Senior
│   ├── detection_agent.py      ← Senior
│   ├── priority_agent.py       ← Junior
│   ├── explainability_agent.py ← Junior
│   └── response_agent.py       ← Junior
├── dashboard/
│   └── app.py                  ← Junior
├── utils/
│   ├── logger.py               ← Junior
│   ├── emailer.py              ← Junior
│   └── groq_summary.py         ← Junior
├── backend/
│   ├── api.py                  ← Senior
│   └── database.py             ← Senior
├── models/                     ← Senior
├── requirements.txt
└── README.md                   ← Junior

## 🚀 Getting Started
```bash
# Clone the repository
git clone https://github.com/Adithya-Yadav18/cyber-threat-detection.git
cd cyber-threat-detection

# Create and activate conda environment
conda create -n cyber-threat python=3.11 -y
conda activate cyber-threat

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dashboard Features
- Live threat feed (auto-refreshes every 3 seconds)
- Attack type breakdown (pie chart + bar chart)
- Severity distribution
- SHAP explanation per threat
- AI-generated incident summaries
- Email alert log
- Live / Simulation mode toggle

## 🔄 Current Status
> Project is complete and deployed. Live dashboard available at https://cybershield-agent.streamlit.app
