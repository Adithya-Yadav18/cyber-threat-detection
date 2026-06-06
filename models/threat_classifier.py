# models/threat_classifier.py
# CyberShield AI - Unified Threat Classifier
# Combines DNN + XGBoost + LSTM using ensemble voting

import tensorflow as tf
import numpy as np
import pickle
import os

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")

class ThreatClassifier:
    """
    Unified threat classifier combining DNN, XGBoost, and LSTM.
    Uses weighted ensemble voting for final prediction.
    """

    def __init__(self):
        self.dnn_model = None
        self.xgb_model = None
        self.lstm_model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.is_loaded = False

    def load_models(self):
        """Load all saved models and preprocessors."""
        print("[*] Loading CyberShield AI models...")
        try:
            # Load DNN
            self.dnn_model = tf.keras.models.load_model(
                os.path.join(MODEL_DIR, "dnn_model.h5")
            )
            print("[+] DNN model loaded.")

            # Load XGBoost
            with open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "rb") as f:
                self.xgb_model = pickle.load(f)
            print("[+] XGBoost model loaded.")

            # Load LSTM
            self.lstm_model = tf.keras.models.load_model(
                os.path.join(MODEL_DIR, "lstm_model.h5")
            )
            print("[+] LSTM model loaded.")

            # Load preprocessors
            with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
                self.scaler = pickle.load(f)

            with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
                self.label_encoder = pickle.load(f)

            with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb") as f:
                self.feature_names = pickle.load(f)

            print("[+] Preprocessors loaded.")
            print(f"[+] Attack classes: {list(self.label_encoder.classes_)}")
            self.is_loaded = True
            print("[✓] All models ready.")

        except Exception as e:
            print(f"[!] Error loading models: {e}")
            raise

    def preprocess_input(self, features: dict) -> np.ndarray:
        """
        Preprocess raw feature dict into model-ready array.
        features: dict with feature names as keys
        """
        # Build feature vector in correct order
        vector = []
        for fname in self.feature_names:
            vector.append(float(features.get(fname, 0.0)))

        X = np.array(vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return X_scaled

    def predict(self, features: dict) -> dict:
        """
        Run ensemble prediction on a single traffic sample.
        Returns full threat report dict.
        """
        if not self.is_loaded:
            self.load_models()

        X = self.preprocess_input(features)

        # ── DNN Prediction ───────────────────────────────────────────────────
        dnn_probs = self.dnn_model.predict(X, verbose=0)[0]
        dnn_class = np.argmax(dnn_probs)
        dnn_confidence = float(dnn_probs[dnn_class])

        # ── XGBoost Prediction ───────────────────────────────────────────────
        xgb_probs = self.xgb_model.predict_proba(X)[0]
        xgb_class = np.argmax(xgb_probs)
        xgb_confidence = float(xgb_probs[xgb_class])

        # ── LSTM Prediction ──────────────────────────────────────────────────
        X_lstm = X[:, :78]
        lstm_probs = self.lstm_model.predict(X_lstm, verbose=0)[0]
        lstm_class = np.argmax(lstm_probs)
        lstm_confidence = float(lstm_probs[lstm_class])

        # ── Weighted Ensemble Voting ─────────────────────────────────────────
        # Weights based on model accuracy: XGB=0.4, DNN=0.35, LSTM=0.25
        ensemble_probs = (
            0.35 * dnn_probs +
            0.40 * xgb_probs +
            0.25 * lstm_probs
        )
        final_class = np.argmax(ensemble_probs)
        final_confidence = float(ensemble_probs[final_class])
        attack_type = self.label_encoder.classes_[final_class]

        # ── Severity Scoring ─────────────────────────────────────────────────
        severity = self._get_severity(attack_type, final_confidence)

        # ── Build Result ─────────────────────────────────────────────────────
        result = {
            "attack_type": attack_type,
            "confidence": round(final_confidence * 100, 2),
            "severity": severity,
            "is_threat": attack_type != "BENIGN",
            "model_votes": {
                "dnn": {
                    "prediction": self.label_encoder.classes_[dnn_class],
                    "confidence": round(dnn_confidence * 100, 2)
                },
                "xgboost": {
                    "prediction": self.label_encoder.classes_[xgb_class],
                    "confidence": round(xgb_confidence * 100, 2)
                },
                "lstm": {
                    "prediction": self.label_encoder.classes_[lstm_class],
                    "confidence": round(lstm_confidence * 100, 2)
                }
            },
            "all_probabilities": {
                self.label_encoder.classes_[i]: round(float(ensemble_probs[i]) * 100, 2)
                for i in range(len(self.label_encoder.classes_))
            }
        }
        return result

    def _get_severity(self, attack_type: str, confidence: float) -> str:
        """Assign severity level based on attack type and confidence."""
        severity_map = {
            "BENIGN":      "SAFE",
            "DDoS":        "CRITICAL",
            "DoS":         "CRITICAL",
            "Bot":         "HIGH",
            "BruteForce":  "HIGH",
            "PortScan":    "MEDIUM",
            "WebAttack":   "HIGH",
            "Infiltration":"CRITICAL",
            "Heartbleed":  "CRITICAL",
            "Other":       "MEDIUM",
        }
        base_severity = severity_map.get(attack_type, "MEDIUM")

        # Downgrade if confidence is low
        if confidence < 0.6 and base_severity == "CRITICAL":
            return "HIGH"
        if confidence < 0.5:
            return "MEDIUM"

        return base_severity

    def predict_batch(self, samples: list) -> list:
        """Run prediction on a list of feature dicts."""
        return [self.predict(s) for s in samples]


# ── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    classifier = ThreatClassifier()
    classifier.load_models()

    # Load feature names for test
    with open("models/saved_model/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    # Simulate a test sample with random values
    import random
    test_sample = {fname: random.uniform(0, 100) for fname in feature_names}

    print("\n[*] Running test prediction...")
    result = classifier.predict(test_sample)

    print("\n" + "="*50)
    print("  THREAT ANALYSIS RESULT")
    print("="*50)
    print(f"  Attack Type  : {result['attack_type']}")
    print(f"  Confidence   : {result['confidence']}%")
    print(f"  Severity     : {result['severity']}")
    print(f"  Is Threat    : {result['is_threat']}")
    print("\n  Model Votes:")
    for model, vote in result['model_votes'].items():
        print(f"    {model:10} → {vote['prediction']:12} ({vote['confidence']}%)")
    print("="*50)