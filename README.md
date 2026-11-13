# 📊 Social Media Impact on Life Classification

A Machine Learning classification project that predicts the **overall impact of social media on a student's life**  **Positive**, **Neutral**, or **Negative** based on survey data (usage habits, stress, GPA, platform, gender, etc.).

The project trains and compares **four classification models** Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors  and evaluates each one with accuracy, precision, recall, F1 score, and a confusion matrix.

---

## ✨ What This Project Does

- 📥 Loads the survey dataset from a CSV file
- 🧹 Cleans the data (fills missing values, drops the `Student_ID` column)
- 🔢 Encodes categorical columns (Gender, Academic Level, Platform, Device, Comparison Frequency) using `LabelEncoder`
- 🎯 Encodes the target class (`Overall_Impact`)
- ✂️ Splits the data into 80% train / 20% test with **stratification**
- 🧠 Trains and evaluates **4 ML models**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors
- 📏 Reports **Accuracy, Precision, Recall, F1 Score** and a **classification report** for each model
- 📊 Plots a **confusion matrix heatmap** for each model
- 📈 Produces a **bar chart** comparing all models side by side

---

## 🧩 Requirements

- **Python 3.10+**
- Packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`



