"""
Rice Quality Prediction - Machine Learning Project
Author: Praveen Nagaraj Nayak
Description: Predict rice quality grade using ML classification models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
def load_data(filepath="data/rice_quality.csv"):
    print("\n📂 Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}")
    return df

# ── 2. EXPLORE DATA ──────────────────────────────────────────────────────────
def explore(df):
    print("\n🔍 Exploratory Data Analysis:")
    print("=" * 50)
    print(df.head())
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nClass Distribution:\n", df["Grade"].value_counts())

    # Plot class distribution
    plt.figure(figsize=(7, 4))
    df["Grade"].value_counts().plot(kind="bar", color=["steelblue","coral","seagreen"], edgecolor="black")
    plt.title("Rice Grade Distribution", fontweight="bold")
    plt.xlabel("Grade")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("outputs/grade_distribution.png", dpi=100)
    plt.close()
    print("   📊 Saved: outputs/grade_distribution.png")

# ── 3. PREPROCESS ────────────────────────────────────────────────────────────
def preprocess(df):
    print("\n🧹 Preprocessing data...")

    le = LabelEncoder()
    df["Grade_Encoded"] = le.fit_transform(df["Grade"])

    feature_cols = ["Length", "Width", "Area", "Perimeter",
                    "Roundness", "Aspect_Ratio", "Color_Score"]
    X = df[feature_cols]
    y = df["Grade_Encoded"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Save scaler and label encoder
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(le,     "models/label_encoder.pkl")

    print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train_sc, X_test_sc, y_train, y_test, le

# ── 4. TRAIN MODELS ──────────────────────────────────────────────────────────
def train_models(X_train, y_train):
    print("\n🤖 Training Models...")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM":                 SVC(kernel="rbf", probability=True, random_state=42),
    }

    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        model.fit(X_train, y_train)
        results[name] = {
            "model":    model,
            "cv_mean":  cv_scores.mean(),
            "cv_std":   cv_scores.std(),
        }
        print(f"   ✅ {name:25s} CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return results

# ── 5. EVALUATE ──────────────────────────────────────────────────────────────
def evaluate(results, X_test, y_test, le):
    print("\n📊 Evaluation on Test Set:")
    print("=" * 50)

    best_name, best_acc = None, 0
    summary = []

    for name, info in results.items():
        model = info["model"]
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        summary.append({"Model": name, "CV Accuracy": round(info["cv_mean"], 4),
                         "Test Accuracy": round(acc, 4)})

        print(f"\n  🔹 {name}")
        print(f"     Test Accuracy : {acc:.4f}")
        print(classification_report(y_test, y_pred,
              target_names=le.classes_, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name} — Confusion Matrix", fontweight="bold")
        plt.tight_layout()
        fname = name.replace(" ", "_").lower()
        plt.savefig(f"outputs/{fname}_confusion.png", dpi=100)
        plt.close()

        if acc > best_acc:
            best_acc, best_name = acc, name

    # Save summary
    summary_df = pd.DataFrame(summary)
    print("\n📋 Model Comparison:")
    print(summary_df.to_string(index=False))
    summary_df.to_csv("outputs/model_comparison.csv", index=False)

    # Bar chart comparison
    plt.figure(figsize=(8, 4))
    plt.bar(summary_df["Model"], summary_df["Test Accuracy"],
            color=["steelblue","coral","seagreen","orchid"], edgecolor="black")
    plt.title("Model Accuracy Comparison", fontweight="bold")
    plt.ylabel("Test Accuracy")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=15, ha="right")
    for i, v in enumerate(summary_df["Test Accuracy"]):
        plt.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png", dpi=100)
    plt.close()
    print("   📊 Saved: outputs/model_comparison.png")

    return best_name, results[best_name]["model"]

# ── 6. SAVE BEST MODEL ───────────────────────────────────────────────────────
def save_best_model(name, model):
    path = "models/best_model.pkl"
    joblib.dump(model, path)
    print(f"\n🏆 Best Model : {name}")
    print(f"💾 Saved to   : {path}")

# ── 7. FEATURE IMPORTANCE ────────────────────────────────────────────────────
def feature_importance(results):
    rf = results["Random Forest"]["model"]
    features = ["Length", "Width", "Area", "Perimeter",
                "Roundness", "Aspect_Ratio", "Color_Score"]
    importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)

    plt.figure(figsize=(7, 4))
    importance.plot(kind="barh", color="steelblue", edgecolor="black")
    plt.title("Feature Importance — Random Forest", fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=100)
    plt.close()
    print("   📊 Saved: outputs/feature_importance.png")

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import generate_data as gd
    gd.create_dataset()

    df = load_data()
    explore(df)
    X_train, X_test, y_train, y_test, le = preprocess(df)
    results = train_models(X_train, y_train)
    best_name, best_model = evaluate(results, X_test, y_test, le)
    save_best_model(best_name, best_model)
    feature_importance(results)

    print("\n🎉 Project complete! Check /outputs and /models folders.")
