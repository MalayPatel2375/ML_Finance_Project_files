import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import(
    confusion_matrix,
    classification_report,
    accuracy_score
)

from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/model_input/model_ready_data.csv")
df = df.drop(columns=["Unnamed: 0"])

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

X = df.drop(columns=["Date", "Target"])
y = df["Target"]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)

print("\nXGBoost Accuracy")
print("----------------")
print(round(accuracy, 4))

print("\nConfusion Matrix")
print("---------------")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report")
print("-------------")
print(classification_report(y_test, predictions))

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(by="Importance", ascending=False)

print("\nTop Features")
print("------------")
print(importance_df.head(10))

print("\nMODEL COMPARISON")
print("------------------")

print("Logistic Regression: ~0.53")

print("Random Forest: ~0.53")

print("XGBoost:", round(accuracy, 4))

train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))

print("\nOverfitting Check")
print("-------------")
print("Train Accuracy:", round(train_accuracy, 4))
print("Test Accuracy:", round(accuracy, 4))

importance_df.to_csv("results/xgboost_feature_importance.csv", index=False)