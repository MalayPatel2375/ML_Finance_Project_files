import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss
)

from xgboost import XGBClassifier

features = [

    "Daily_Return",

    "Return_Lag1",

    "Return_Lag2",

    "Return_Lag3",

    "Volatility_Ratio",

    "Volatility_Trend",

    "Return_Volume",

    "VolRation_Volume"
]

df = pd.read_csv("data/model_input/model_interaction_data.csv")

target = "Target"

X = df[features]
y = df[target]

split_index = int(len(df)*0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBClassifier(
    n_estimator = 50,
    random_state = 42,
    max_depth = 2,
    learning_rate = 0.01
)

model.fit(X_train_scaled, y_train)

probabilities = model.predict_proba(X_test_scaled)[:,1]
predictions = (probabilities >= 0.5).astype(int)

print("\nProbability Summary")
print("-----------------------")

print(pd.Series(probabilities).describe())

brier = brier_score_loss(y_test, probabilities)

print("\nBRIER SCORE")
print("-------------------")

print(round(brier, 4))

confidence = np.maximum(probabilities, 1-probabilities)

print("\nCONFIDENCE SUMMARY")
print("-------------------")

print(pd.Series(confidence).describe())

buckets = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

for threshold in buckets:
    mask = confidence >= threshold
    trade_count = mask.sum()

    if trade_count == 0:
        print(f"\nThreshold: {threshold}")
        print("No Signals")
        continue

    accuracy = accuracy_score(y_test[mask], predictions[mask])

    print(f"\nThreshold: {threshold}")
    print(f"\nSignals: {trade_count}")
    print(f"\nAccuracy: {accuracy:.4f}")

print("\nProbability Range")
print("------------------")

print(f"Minimum: {round(probabilities.min(), 4)}")
print(f"Maximum: {round(probabilities.max(), 4)}")
print(f"Average: {round(probabilities.mean(), 4)}")

high_confidence = confidence >= 0.6

print("\nHigh Confidence Signals")
print("----------------------------")
print(high_confidence.sum())








##  DAY 3  ##
results_df = X_test.copy()
results_df['Actual'] = y_test.values
results_df['Predictions'] = predictions
results_df['Probability'] = probabilities
results_df['Confidence'] = confidence

threshold = 0.55

high_conf_df = results_df[results_df['Confidence'] >= threshold].copy()

print("HIGH CONFIDENCE SUMMARY")
print("---------------------")
print(f"Signals: {len(high_conf_df)}")
print(f"Percentage of test set: {round((len(high_conf_df) / len(results_df))*100, 2)}%")

accuracy_2 = accuracy_score(high_conf_df['Actual'], high_conf_df['Predictions'])

print(f"Accuracy: {round(accuracy_2, 4)}")

print("Prediction Distribution")
print("------------------")
print(high_conf_df['Predictions'].value_counts())

print("Actual Distribution")
print("------------------")
print(high_conf_df['Actual'].value_counts())

print("Average Confidence")
print("-----------------")
print(round(high_conf_df['Confidence'].mean(), 4))

top_preds = high_conf_df.sort_values("Confidence", ascending=False).head(10)

print("TOP PREDICTIONS")
print("-----------------------")
print(top_preds[['Confidence', 'Probability', 'Predictions', 'Actual']])

plt.figure(figsize=(8, 5))

plt.hist(confidence, bins=20, edgecolor="black")

plt.title("Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.show()

