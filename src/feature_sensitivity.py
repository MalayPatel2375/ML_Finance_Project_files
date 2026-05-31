import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

df = pd.read_csv("data/model_input/model_ready_data.csv")

df = df.drop(columns=["Unnamed: 0"])

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

selected_features = [

    "Daily_Return",

    "RSI_14",

    "Momentum_5",

    "Volatility_5",

    "Return_Lag1",

    "Return_Lag2",

    "Return_Lag3",

    "Price_VS_SMA5",

    "Price_VS_SMA20"
]

y = df["Target"]

results = []

for removed_feature in selected_features:
    current_features = [
        feature for feature in selected_features if feature != removed_feature
    ]

    X = df[current_features]

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]

    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]

    y_test = y.iloc[split_index:]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = XGBClassifier(

        n_estimators=50,

        max_depth=2,

        learning_rate=0.01,

        random_state=42
    )

    model.fit(

        X_train_scaled,

        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(

        y_test,

        predictions
    )

    results.append({

    "Removed_Feature":
    removed_feature,

    "Accuracy":
    accuracy

    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(by="Accuracy", ascending=True)

print("\nFeature Sensitivity Results")
print("----------------------")
print(results_df)

baseline_accuracy = 0.5355

results_df["Accuracy_Change"] = results_df["Accuracy"] - baseline_accuracy

print("\nAccuracy Change")
print("---------------")

print(results_df[["Removed_Feature", "Accuracy", "Accuracy_Change"]])

results_df.to_csv("results/feature_sensitivity.csv", index=False)