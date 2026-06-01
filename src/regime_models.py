import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

df = pd.read_csv("data/model_input/model_ready_data.csv")

df = df.drop(columns=["Unnamed: 0"])

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

df["Market_Regime"] = np.where(df["Daily_Return"] >= 0, "Bullish", "Bearish")

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

results = []

for regime in ["Bullish", "Bearish"]:
    regime_df = df[df["Market_Regime"] == regime]

    X = regime_df[selected_features]
    y = regime_df["Target"]

    split_index = int(len(regime_df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(
    n_estimators=50,
    max_depth=2,
    learning_rate=0.01,
    random_state=42)

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)

    results.append({
        "Regime": regime,
        "Samples": len(regime_df),
        "Accuracy": accuracy
    })

    print(f"\n{regime.upper()} MODEL")
    print("------------------")

    print(f"Accuracy: {round(accuracy, 4)}")
    print(f"Samples: {len(regime_df)}")

results_df = pd.DataFrame(results)

print("\nREGIME MODEL COMPARISONS")
print("--------------------")
print(results_df)

results_df.to_csv("results/regime_model_results.csv", index=False)