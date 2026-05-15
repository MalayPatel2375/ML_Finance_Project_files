import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

#_________________________________________________________________________________________________________________________
# Logistic Regression

df = pd.read_csv("data/model_input/model_ready_data.csv")
df = df.drop(columns=["Unnamed: 0"])

#print(df.shape)
#print(df.head())

X = df.drop(columns=["Target"])
y = df["Target"]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, predictions)
print(f"\nConfusion Matrix: {cm}")

print("\nClassification Report: ")
print(classification_report(y_test, predictions))

feature_importance = pd.DataFrame({
    "Features": X.columns,
    "Coefficients": model.coef_[0]
})

print(feature_importance.sort_values(by="Coefficients", ascending=False))

#________________________________________________________________________________________________________________________
# Random Forest
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f}")

rf_cm = confusion_matrix(y_test, rf_predictions)

print("\nRandom Forest Confusion Matrix:")
print(rf_cm)

print("\nRandom Forest Classification Report:")

print(classification_report(y_test, rf_predictions))

rf_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

print(
    rf_importance.sort_values(
        by="Importance",
        ascending=False
    )
)

#____________________________________________________________________________________________________________________
# Comparing Models

print("\nMODEL COMPARISON")
print("----------------")
print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")


train_accuracy = rf_model.score(X_train, y_train)

test_accuracy = rf_model.score(X_test, y_test)

print("\nOverfitting Check")
print("------------------")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

rf_importance.sort_values(by="Importance", ascending=False).to_csv("results/feature_importance.csv", index=False)