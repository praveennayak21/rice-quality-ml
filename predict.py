"""
Predict rice quality grade for new samples
Author: Praveen Nagaraj Nayak
Usage: python predict.py
"""

import joblib
import numpy as np
import pandas as pd

def predict(sample: dict) -> str:
    scaler = joblib.load("models/scaler.pkl")
    le     = joblib.load("models/label_encoder.pkl")
    model  = joblib.load("models/best_model.pkl")

    features = ["Length", "Width", "Area", "Perimeter",
                "Roundness", "Aspect_Ratio", "Color_Score"]

    X = pd.DataFrame([sample])[features]
    X_sc = scaler.transform(X)

    pred_encoded = model.predict(X_sc)[0]
    pred_label   = le.inverse_transform([pred_encoded])[0]
    proba        = model.predict_proba(X_sc)[0]

    print("\n🌾 Rice Quality Prediction")
    print("=" * 35)
    for k, v in sample.items():
        print(f"  {k:15s}: {v}")
    print(f"\n  🏆 Predicted Grade : {pred_label}")
    print(f"  📊 Confidence      : {max(proba)*100:.1f}%")
    print("\n  Class Probabilities:")
    for cls, p in zip(le.classes_, proba):
        bar = "█" * int(p * 20)
        print(f"    {cls}: {bar:20s} {p*100:.1f}%")

    return pred_label

if __name__ == "__main__":
    # Example: predict for a new rice sample
    sample = {
        "Length":       7.3,
        "Width":        3.4,
        "Area":         25.1,
        "Perimeter":    21.5,
        "Roundness":    0.83,
        "Aspect_Ratio": 2.15,
        "Color_Score":  0.88,
    }
    predict(sample)
