import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/model_input/model_ready_data.csv")
df = df.drop(columns=["Unnamed: 0"])

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

X = df.drop(
    columns=["Date", "Target"]
)

y = df["Target"]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

acccuracy = accuracy_score(y_test, predictions)

print("\nTime Series Validation Accuracy")
print("------------------------")

print(round(acccuracy, 4))

print("\nConfusion Matrix")
print("--------------")

print(confusion_matrix(y_test, predictions))

print("\nClassification Report")
print("-----------------")

print(classification_report(y_test, predictions))

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

results.to_csv("results/time_series_preds.csv", index=False)
