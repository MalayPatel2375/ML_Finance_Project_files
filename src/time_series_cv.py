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

tscv = TimeSeriesSplit(n_splits=5)

scores = []

fold = 1

for train_index, test_index in tscv.split(X):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

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

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)

    scores.append(accuracy)

    print(f"\nFold {fold}")
    print("----------------")
    print("Accuracy:",round(accuracy, 4))

    fold += 1

average_score = np.mean(scores)
std_score = np.std(scores)

print("\nFINAL CROSS-VALIDATION RESULTS")
print("--------------------------------")

print("Average Accuracy:",round(average_score, 4))
print("Standard Deviation:",round(std_score, 4))

results_df = pd.DataFrame({
    "Fold": range(1, 6),
    "Accuracy": scores
})

results_df.to_csv("results/time_series_cv_result.csv", index=False)
