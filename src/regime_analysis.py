import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

df = pd.read_csv("data/model_input/model_ready_data.csv")

df = df.drop(columns=["Unnamed: 0"])
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")
#print(df["Price_VS_SMA20"].describe())

volatility_threshold = df["Volatility_5"].median()

df["Volatility_Regime"] = np.where(df["Volatility_5"] >= volatility_threshold, "High_volatility", "Low_volatility")

df["Trend_Regime"] = np.where(df["Price_VS_SMA20"] >= 0, "Bullish", "Bearish")

X = df.drop(columns=["Target", "Date", "Volatility_Regime", "Trend_Regime"])
y = df["Target"]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

test_volatiltiy_regime = df["Volatility_Regime"].iloc[split_index:]
test_trend_regime = df["Trend_Regime"].iloc[split_index:]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBClassifier(
    n_estimators = 50,
    random_state = 42,
    learning_rate = 0.01,
    max_depth = 2
)

model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predictions": predictions,
    "Volatility_Regime": test_volatiltiy_regime.values,
    "Trend_Regime": test_trend_regime.values
})

print("\nVOLATILITY REGIME ANALYSIS")
print("--------------------------------")

for regime in results_df["Volatility_Regime"].unique():
    regime_df = results_df[results_df["Volatility_Regime"] == regime]
    
    regime_accuracy = accuracy_score(regime_df["Actual"], regime_df["Predictions"])

    print(f"\n{regime}")
    print(f"Accuracy: {round(regime_accuracy, 4)}")
    print(f"Samples: {len(regime_df)}")

print("\nTREND REGIME ANALYSIS")
print("---------------------")

for regime in results_df["Trend_Regime"].unique():
    regime_df = results_df[results_df["Trend_Regime"] == regime]
    
    regime_accuracy = accuracy_score(regime_df["Actual"], regime_df["Predictions"])

    print(f"\n{regime}")
    print(f"Accuracy: {round(regime_accuracy, 4)}")
    print(f"Samples: {len(regime_df)}")

results_df.to_csv("results/regime_analysis.csv", index=False)
