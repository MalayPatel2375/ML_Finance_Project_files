import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    brier_score_loss
)

df = pd.read_csv("data/model_input/model_ready_data.csv")

df = df.drop(columns=["Unnamed: 0"])

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

selected_features = [
    "Daily_Return", "RSI_14", "Momentum_5", "Volatility_5",
    "Return_Lag1", "Return_Lag2", "Return_Lag3", "Price_VS_SMA5", "Price_VS_SMA20"
]

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

model = XGBClassifier(
    n_estimators = 50,
    random_state = 42,
    max_depth = 2,
    learning_rate = 0.01
)

model.fit(X_train_scaled, y_train)

probabilities = model.predict_proba(X_test_scaled)

up_probabilities = probabilities[:, 1]

predictions = np.where(up_probabilities >= 0.5, 1, 0)

accuracy = accuracy_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("---------------------")

print("Accuracy:", round(accuracy, 4))

brier = brier_score_loss(y_test, up_probabilities)

print("\nBrier Score")
print("---------------------")

print("Brier Score:", round(brier, 4))

print("\nConfidence Analysis")
print("-------------------")

mean_confidence = up_probabilities.mean()

print("Average Bullish Probabilities: ", round(mean_confidence, 4))
print(f"Maximum Bullish Probability: {round(up_probabilities.max(), 4)}")
print(f"Minimum Bullish Probability: {round(up_probabilities.min(), 4)}")

high_confidence = up_probabilities >= 0.7

high_confidence_count = high_confidence.sum()

print(f"\nHigh Confidence Predictions: {high_confidence_count}")

if high_confidence_count > 0:

    high_confi_accuracy = accuracy_score(y_test[high_confidence], predictions[high_confidence])

    print(f"High Confidence Accuracy: {round(high_confi_accuracy, 4)}")

results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Prediction": predictions,
    "Bullish Probability": up_probabilities
})

results_df.to_csv("results/probability_analysis.csv", index=False)
