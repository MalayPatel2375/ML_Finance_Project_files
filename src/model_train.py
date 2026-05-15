import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

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

