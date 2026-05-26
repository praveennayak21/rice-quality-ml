# 🌾 Rice Quality Prediction — Machine Learning

A Machine Learning project that predicts rice quality grade (Grade A, B, C) using classification algorithms including Logistic Regression, Decision Tree, Random Forest, and SVM.

**Built by:** [Praveen Nagaraj Nayak](https://github.com/praveennayak21)

---

## ✨ Features

- 🤖 4 ML models compared — Logistic Regression, Decision Tree, Random Forest, SVM
- 📊 Cross-validation with accuracy scores
- 📈 Confusion matrix and model comparison charts
- 🌾 Feature importance analysis
- 💾 Best model saved automatically using joblib
- 🔮 Predict grade for new rice samples

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

---

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/praveennayak21/rice-quality-ml.git
cd rice-quality-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train all models
```bash
python train.py
```

### 4. Predict for a new sample
```bash
python predict.py
```

---

## 📁 Project Structure

```
rice-quality-ml/
├── train.py             # Train and evaluate all ML models
├── predict.py           # Predict grade for new rice sample
├── generate_data.py     # Generate synthetic rice dataset
├── requirements.txt     # Python dependencies
├── .gitignore
├── README.md
├── data/
│   └── rice_quality.csv # Dataset (auto-generated)
├── models/
│   ├── best_model.pkl   # Saved best model
│   ├── scaler.pkl       # Feature scaler
│   └── label_encoder.pkl
└── outputs/
    ├── grade_distribution.png
    ├── feature_importance.png
    ├── model_comparison.png
    └── *_confusion.png
```

---

## 📊 Models Compared

| Model | Description |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Rule-based tree classifier |
| Random Forest | Ensemble of 100 trees |
| SVM | Support Vector Machine with RBF kernel |

---

## 🔮 Sample Prediction Output

```
🌾 Rice Quality Prediction
===================================
  Length        : 7.3
  Width         : 3.4
  Area          : 25.1
  Roundness     : 0.83
  Color_Score   : 0.88

  🏆 Predicted Grade : Grade A
  📊 Confidence      : 94.2%
```

---

## 🔗 Connect

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/praveennayak21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/praveen-n-nayak-36b92330b)
