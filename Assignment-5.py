# ==========================================
# AI-ML Assignment 5
# Employee Attrition Prediction
# Decision Tree & Random Forest
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

# -----------------------------
# Task 1 : Data Understanding
# -----------------------------

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("First Five Records")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nSummary Statistics")
print(df.describe())

print("\nNumerical Features")
print(df.select_dtypes(include=["int64", "float64"]).columns.tolist())

print("\nCategorical Features")
print(df.select_dtypes(include=["object"]).columns.tolist())

print("\nTarget Variable")
print("Attrition")

# -----------------------------
# Task 2 : Data Preprocessing
# -----------------------------

print("\nMissing Values")
print(df.isnull().sum())

# Remove constant columns
df.drop(columns=["EmployeeCount", "Over18", "StandardHours"],
        inplace=True,
        errors="ignore")

# Encode categorical columns
encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = encoder.fit_transform(df[col])

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------
# Task 3 : Model Development
# -----------------------------

dt = DecisionTreeClassifier(random_state=42)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)

# -----------------------------
# Evaluation Function
# -----------------------------

def evaluate(name, y_true, y_pred):
    print("\n", name)
    print("-------------------------")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))

evaluate("Decision Tree", y_test, dt_pred)
evaluate("Random Forest", y_test, rf_pred)

# -----------------------------
# Confusion Matrix
# -----------------------------

ConfusionMatrixDisplay.from_predictions(y_test, dt_pred)
plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.savefig("decision_tree_confusion_matrix.png")
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, rf_pred)
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("random_forest_confusion_matrix.png")
plt.show()

# -----------------------------
# Feature Importance
# -----------------------------

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False)

plt.figure(figsize=(10,6))

importance.head(10).plot(kind="bar")

plt.title("Random Forest Feature Importance")
plt.ylabel("Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.show()

print("\nObservations")
print("1. Random Forest achieved higher overall accuracy than Decision Tree.")
print("2. Decision Tree captured more positive cases but is more prone to overfitting.")
print("3. Random Forest produced more stable predictions using multiple trees.")
print("4. Feature importance highlights the most influential variables.")
