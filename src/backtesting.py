import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/model_input/model_ready_data.csv")

if "Unamed: 0" in df.columns:
    df = df.drop(columns=["Unamed: 0"])

X = df.drop(columns=["Target"])
y = df["Target"]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()

model.fit(X_train_scaled, y_train)

probabilities = model.predict_proba(X_test_scaled)

up_probabilities = probabilities[:, 1]

print(up_probabilities[:20])
print("\nMax Probability: ", up_probabilities.max())
print("Min Probability: ", up_probabilities.min())
print("Mean Probability: ", up_probabilities.mean())
print("\n")

signals = np.where(up_probabilities > 0.50, 1, 0) #threshhold tuned from 0.55 to 0.51

test_df = df.iloc[split_index:].copy()

test_df["Strategy_Signal"] = signals

test_df["Market_Return"] = test_df["Daily_Return"]

test_df["Strategy_Return"] = (test_df["Strategy_Signal"] * test_df["Market_Return"])

test_df["Probability_Up"] = up_probabilities

avg_market_return = test_df["Market_Return"].mean()

avg_strategy_return = (
    test_df.loc[
        test_df["Strategy_Signal"] == 1,
        "Market_Return"
    ].mean()
)

print("\nAVERAGE RETURNS")
print("------------------")

print(f"Average Market Return: {avg_market_return:.6f}")

print(f"Average Strategy Return: {avg_strategy_return:.6f}")


print("\nTrade Count")
print("------------")

print(test_df["Strategy_Signal"].value_counts())

test_df.to_csv("results/backtest_results.csv", index=False)
