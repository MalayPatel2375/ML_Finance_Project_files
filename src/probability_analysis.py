import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/model_input/model_ready_data.csv")

df = df.drop(columns=["Unnamed: 0"])
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

X = df.drop(columns=["Date", "Target"])
y = df["Target"]

split_index = int(len(df)*0.8)

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

probabilities = model.predict_proba(X_test_scaled)
up_probability = probabilities[:, 1]

predictions = (up_probability >= 0.5).astype(int)

overall_accuracy = accuracy_score(y_test, predictions)

print("\nOverall Accuracy")
print("-----------------")

print(round(overall_accuracy, 4))

results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Prediction": predictions,
    "Up_probability": up_probability
})

results_df["Confidence"] = np.where(
    results_df["Up_probability"] >= 0.50,
    results_df["Up_probability"],
    1 - results_df["Up_probability"]
)

thresholds = [0.50, 0.60, 0.70, 0.80]

print("\nCONFIDENCE THRESHOLD ANALYSIS")
print("--------------------------------")

for threshold in thresholds:
    filtered_df = results_df[results_df["Confidence"] >= threshold]

    if len(filtered_df) == 0:
        print(f"\nThreshold: {threshold}")
        print("No Trades Found")
        continue

    accuracy = accuracy_score(filtered_df["Actual"], filtered_df["Prediction"])

    print(f"\nThreshold: {threshold}")
    print(f"Accuracy: {round(accuracy, 4)}")
    print(f"Trade Count: {len(filtered_df)}")

results_df.to_csv("results/prob_analysis_results.csv", index=False)
