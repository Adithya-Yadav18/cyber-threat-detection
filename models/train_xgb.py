# models/train_xgb.py
# CyberShield AI - XGBoost Classifier
# Fast, accurate gradient boosting for threat detection

import xgboost as xgb
import numpy as np
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import accuracy_score, classification_report
from utils.preprocessor import run_preprocessing

def train_xgb():
    """Full XGBoost training pipeline."""
    print("=" * 60)
    print("  CyberShield AI - XGBoost Training")
    print("=" * 60)

    # Load preprocessed data
    X_train, X_test, y_train, y_test, features, le = run_preprocessing()

    num_classes = len(le.classes_)
    print(f"\n[*] Training XGBoost model...")
    print(f"    Classes: {list(le.classes_)}")
    print(f"    Train size: {len(X_train):,}")

    # Build XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )

    # Train
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # Evaluate
    print("\n[*] Evaluating XGBoost model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[+] Test Accuracy: {accuracy * 100:.2f}%")
    print("\n[+] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model
    with open('models/saved_model/xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("[+] XGBoost model saved to models/saved_model/xgb_model.pkl")

    # Save feature importances
    importances = model.feature_importances_
    feature_importance_dict = dict(zip(features, importances))
    sorted_features = dict(sorted(
        feature_importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )[:20])

    with open('models/saved_model/feature_importances.pkl', 'wb') as f:
        pickle.dump(sorted_features, f)
    print("[+] Top 20 feature importances saved.")
    print("\n[✓] XGBoost Training Complete!")

if __name__ == "__main__":
    train_xgb()