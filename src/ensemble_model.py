import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import(
    accuracy_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("data/model_input/model_ready_data.csv")

df = df.drop(columns=["Unnamed: 0"])
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

selected_features = ["Daily_Return", "RSI_14", "Momentum_5", "Volatility_5", "Return_Lag1",
                     "Return_Lag2", "Return_Lag3", "Price_VS_SMA5", "Price_VS_SMA20"]

X = df[selected_features]
y = df["Target"]

split_index = int(len(df) * 0.8)
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

xgb_model = XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.01, random_state=42)

lr_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train_scaled, y_train)
xgb_model.fit(X_train_scaled, y_train)

lr_preds = lr_model.predict(X_test_scaled)
rf_preds = rf_model.predict(X_test_scaled)
xgb_preds = xgb_model.predict(X_test_scaled)

ensemble_predictions = (lr_preds + rf_preds + xgb_preds)

ensemble_predictions = np.where(ensemble_predictions >= 2, 1, 0)

ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)

print("\nENSEMBLE MODEL RESULTS")
print("-----------------------")
print(f"Ensemble Accuracy: {round(ensemble_accuracy, 4)}")

print("\nConfusion Matrix")
print("-----------------------")
print(confusion_matrix(y_test, ensemble_predictions))

print("\nClassification Report")
print("---------------------")
print(classification_report(y_test, ensemble_predictions))

agreement = ((lr_preds == rf_preds) & (rf_preds == xgb_preds))
agreement_rate = agreement.mean()

print("\nAgreement Rate")
print("------------------------")
print("Agreement_Rate: ", round(agreement_rate, 4))

results_df = pd.DataFrame({

    "Actual":
    y_test.values,

    "LR_Prediction":
    lr_preds,

    "RF_Prediction":
    rf_preds,

    "XGB_Prediction":
    xgb_preds,

    "Ensemble_Prediction":
    ensemble_predictions
})

results_df.to_csv("results/ensemble_predictions.csv", index=False)
