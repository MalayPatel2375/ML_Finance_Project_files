import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import shap

from xgboost import XGBClassifier
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
    n_estimators = 50,
    max_depth = 2,
    learning_rate = 0.01,
    random_state = 42
)

model.fit(X_train_scaled, y_train)

explainer = shap.Explainer(model)

shap_values = explainer(X_test_scaled)

shap.summary_plot(shap_values, X_test, show=False)

plt.savefig("results/shap_summary_plot.png", bbox_inches="tight")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Mean_shap_value": np.abs(shap_values.values).mean(axis=0)
})

importance_df = importance_df.sort_values(by="Mean_shap_value", ascending=False)

print("\nSHAP Feature Importance")
print("-------------------")
print(importance_df.head(10))

sample_index = 0

print("\nIndiviual Prediction Analysis")
print("-------------------")

local_values = shap_values.values[sample_index]

local_df = pd.DataFrame({
    "Feature": X.columns,
    "SHAP_contribution": local_values
})

local_df = local_df.sort_values(by="SHAP_contribution", ascending=False)

print(local_df.head(10))

importance_df.to_csv("results/shap_feature_importance.csv", index=False)
local_df.to_csv("results/local_prediction_explanation.csv", index=False)

