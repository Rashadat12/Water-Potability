# -*- coding: utf-8 -*-
"""
Water Potability Prediction
Machine Learning pipeline for predicting water potability using SVM, Decision Tree, and KNN.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    classification_report
)


def load_and_preprocess(filepath="water_potability.csv"):
    print("Loading dataset from:", filepath)
    initial_data = pd.read_csv(filepath)
    print("\n--- Initial Dataset Head ---")
    print(initial_data.head())
    print("\n--- Dataset Info ---")
    print(initial_data.info())
    print("\nDataset Shape:", initial_data.shape)

    # Missing values before imputation
    print("\n--- Missing Values Before Imputation ---")
    print(initial_data.isnull().sum())

    # Impute missing values with most frequent value
    imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    columns_to_impute = ["ph", "Trihalomethanes", "Sulfate"]
    for col in columns_to_impute:
        if col in initial_data.columns:
            initial_data[col] = imputer.fit_transform(initial_data[[col]])

    print("\n--- Missing Values After Imputation ---")
    print(initial_data.isnull().sum())

    # Class balance handling
    print("\n--- Class Distribution (Before Upsampling) ---")
    print(initial_data["Potability"].value_counts())

    not_potable = initial_data[initial_data["Potability"] == 0]
    potable = initial_data[initial_data["Potability"] == 1]

    # Upsample minority class (Potable = 1)
    upsampled_potable = resample(
        potable,
        replace=True,
        n_samples=len(not_potable),
        random_state=42
    )
    balanced_data = pd.concat([not_potable, upsampled_potable])
    balanced_data = shuffle(balanced_data, random_state=42).reset_index(drop=True)

    print("\n--- Class Distribution (After Upsampling) ---")
    print(balanced_data["Potability"].value_counts())

    return balanced_data


def train_and_evaluate(data):
    X = data.drop(["Potability"], axis=1)
    y = data["Potability"]

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\nTrain set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    models = {
        "Support Vector Machine (RBF)": svm.SVC(kernel="rbf", C=1.0, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors (k=3)": KNeighborsClassifier(n_neighbors=3)
    }

    results = {}

    for name, clf in models.items():
        print(f"\n==================== {name} ====================")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Model": clf
        }

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Not Potable (0)", "Potable (1)"]))

    return results, X_test, y_test


if __name__ == "__main__":
    df = load_and_preprocess("water_potability.csv")
    results, X_test, y_test = train_and_evaluate(df)
    print("\n=== Model Comparison Summary ===")
    summary_df = pd.DataFrame(results).T[["Accuracy", "Precision", "Recall", "F1-Score"]]
    print(summary_df)
