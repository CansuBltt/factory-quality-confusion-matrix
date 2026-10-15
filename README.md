# 🏭 Factory Product Quality Classification & Model Interpretability

This project is a Machine Learning classification pipeline that uses 10 sensor features to predict product quality (**Healthy, Scratched, Stained, Broken**) in a manufacturing plant and visualizes model decisions.

---

## 📌 Key Features

* **Stratified Splitting:** Uses `stratify=y` to preserve minority class ratios (Broken, Stained) across train and test sets.
* **Error Analysis (Confusion Matrix):** Evaluates model misclassifications and false positives/negatives using heatmap matrix plots.
* **Model Interpretability (Feature Importance):** Identifies the most impactful sensors influencing quality decisions.

---

## 🛠️ Tech Stack

* **Python 3.x**
* **Scikit-Learn** (`RandomForestClassifier`, `confusion_matrix`, `classification_report`)
* **Matplotlib & Seaborn** (Data Visualization)
* **NumPy & Pandas** (Data Processing)

---

## 🚀 How to Run

1. Install required dependencies:
   ```bash
   python -m pip install numpy pandas matplotlib seaborn scikit-learn