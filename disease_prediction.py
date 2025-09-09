# disease_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer

# -----------------------------
# Load dataset (example: Breast Cancer)
# -----------------------------
def load_breast_cancer_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y

# For diabetes / heart datasets, you’ll read CSV
def load_csv_data(path, target_column):
    df = pd.read_csv(path)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y

# -----------------------------
# Train & evaluate models
# -----------------------------
def train_and_evaluate(X, y):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel="linear", probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    # Train & evaluate
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"\n{name} Results:")
        print("Accuracy:", acc)
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Option 1: Breast Cancer dataset
    X, y = load_breast_cancer_data()

    # Option 2: Replace with CSV dataset (diabetes/heart)
    if __name__ == "__main__":
     X, y = load_csv_data(r"C:\Users\swaya\OneDrive\Desktop\PYTHON\diabetes.csv", "Outcome")
    # X, y = load_csv_data("C:/Users/swaya/OneDrive/Desktop/PYTHON/heart.csv", "target")


    train_and_evaluate(X, y)
