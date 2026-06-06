# agents/explainability_agent.py
# Uses SHAP to explain why a threat was flagged by the ML model
# Junior Developer: Sarvani

import shap
import numpy as np
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────
# SHAP EXPLANATION FUNCTION
# ─────────────────────────────────────────────

def explain_threat(model, input_features: np.ndarray, feature_names: list) -> dict:
    """
    Uses SHAP to explain why the model flagged this input as a threat.

    Args:
        model: Trained ML model (XGBoost or DNN)
        input_features (np.ndarray): Feature values for the detected threat
        feature_names (list): Names of all features

    Returns:
        dict: SHAP explanation containing top contributing features
        Example:
        {
            "top_features": [
                {"feature": "packet_rate", "shap_value": 0.45, "direction": "increases risk"},
                {"feature": "byte_size", "shap_value": 0.30, "direction": "increases risk"},
                {"feature": "port_number", "shap_value": -0.12, "direction": "decreases risk"}
            ],
            "explanation_summary": "This was flagged mainly due to high packet_rate...",
            "timestamp": "2024-01-01 12:00:00"
        }
    """

    try:
        # ── Create SHAP explainer ──
        explainer = shap.Explainer(model, feature_names=feature_names)
        shap_values = explainer(input_features)

        # ── Extract SHAP values for first prediction ──
        values = shap_values.values[0]

        # ── Build feature importance list ──
        feature_importance = []
        for i, name in enumerate(feature_names):
            feature_importance.append({
                "feature": name,
                "shap_value": float(values[i]),
                "direction": "increases risk" if values[i] > 0 else "decreases risk"
            })

        # ── Sort by absolute SHAP value (most important first) ──
        feature_importance.sort(
            key=lambda x: abs(x["shap_value"]),
            reverse=True
        )

        # ── Keep only top 5 features ──
        top_features = feature_importance[:5]

        # ── Build explanation summary ──
        top_name = top_features[0]["feature"] if top_features else "unknown"
        summary = (
            f"This traffic was flagged mainly due to abnormal '{top_name}'. "
            f"Top {len(top_features)} contributing factors identified by SHAP analysis."
        )

        result = {
            "top_features": top_features,
            "explanation_summary": summary,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print(f"[SHAP] Explanation generated — top feature: {top_name}")
        return result

    except Exception as e:
        print(f"[ERROR] SHAP explanation failed: {e}")
        return _fallback_explanation()


# ─────────────────────────────────────────────
# SIMPLE EXPLANATION (for tree models like XGBoost)
# ─────────────────────────────────────────────

def explain_with_tree_explainer(model, input_features: np.ndarray, feature_names: list) -> dict:
    """
    Uses SHAP TreeExplainer specifically for XGBoost model.
    Faster and more accurate for tree-based models.

    Args:
        model: Trained XGBoost model
        input_features (np.ndarray): Feature values for the detected threat
        feature_names (list): Names of all features

    Returns:
        dict: SHAP explanation containing top contributing features
    """

    try:
        # ── Use TreeExplainer for XGBoost ──
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_features)

        # ── Handle multi-class output ──
        if isinstance(shap_values, list):
            values = shap_values[0][0]
        else:
            values = shap_values[0]

        # ── Build feature importance list ──
        feature_importance = []
        for i, name in enumerate(feature_names):
            feature_importance.append({
                "feature": name,
                "shap_value": float(values[i]),
                "direction": "increases risk" if values[i] > 0 else "decreases risk"
            })

        # ── Sort by absolute SHAP value ──
        feature_importance.sort(
            key=lambda x: abs(x["shap_value"]),
            reverse=True
        )

        # ── Keep top 5 features ──
        top_features = feature_importance[:5]

        # ── Build summary ──
        top_name = top_features[0]["feature"] if top_features else "unknown"
        summary = (
            f"This traffic was flagged mainly due to abnormal '{top_name}'. "
            f"Top {len(top_features)} contributing factors identified by SHAP analysis."
        )

        result = {
            "top_features": top_features,
            "explanation_summary": summary,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print(f"[SHAP] Tree explanation generated — top feature: {top_name}")
        return result

    except Exception as e:
        print(f"[ERROR] SHAP TreeExplainer failed: {e}")
        return _fallback_explanation()


# ─────────────────────────────────────────────
# FALLBACK EXPLANATION
# ─────────────────────────────────────────────

def _fallback_explanation() -> dict:
    """
    Returns a basic explanation if SHAP fails.

    Returns:
        dict: Basic fallback explanation
    """
    return {
        "top_features": [
            {"feature": "packet_rate", "shap_value": 0.0, "direction": "unknown"},
            {"feature": "byte_size", "shap_value": 0.0, "direction": "unknown"},
            {"feature": "port_number", "shap_value": 0.0, "direction": "unknown"}
        ],
        "explanation_summary": "SHAP explanation unavailable — model explanation could not be generated.",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }