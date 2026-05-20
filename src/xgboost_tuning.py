import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import (
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

parameter_sets = [

    {
        "n_estimators": 50,
        "max_depth": 2,
        "learning_rate": 0.01
    },

    {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.05
    },

    {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1
    },

    {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.2
    }

]

results = []

for params in parameter_sets:

    print("\nTesting Parameters")
    print("--------------------")

    print(params)

    model = XGBClassifier(
        n_estimators = params["n_estimators"],
        max_depth = params["max_depth"],
        learning_rate = params["learning_rate"],
        random_state = 42
    )

    model.fit(X_train_scaled, y_train)

    train_preds = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_preds)

    test_preds = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, test_preds)

    print(f"Training Accuracy: {round(train_accuracy, 4)}")
    print(f"Testing Accuracy: {round(test_accuracy, 4)}")

    gap = train_accuracy - test_accuracy
    print(f"Overfitting Gap: {round(gap, 4)}")

    results.append({

        "n_estimators":
        params["n_estimators"],

        "max_depth":
        params["max_depth"],

        "learning_rate":
        params["learning_rate"],

        "train_accuracy":
        train_accuracy,

        "test_accuracy":
        test_accuracy,

        "overfitting_gap":
        gap

    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(by="test_accuracy", ascending=False)

print("\nFinal Tuning Results:")
print("----------------")
print(results_df)

results_df.to_csv("results/xgboost_tuning_results.csv", index=False)

