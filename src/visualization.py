import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("results/backtest_results.csv")

df["Cumulative_Market"] = (1 + df["Market_Return"]).cumprod()

df["Cumulative_Strategy"] = (1 + df["Strategy_Return"]).cumprod()

# Plot Equity Curves

plt.figure(figsize=(12,6))

plt.plot(df["Cumulative_Market"], label="Market")

plt.plot(df["Cumulative_Strategy"], label="Strategy")

plt.title("Strategy V/S Market Performance")

plt.xlabel("Time")

plt.ylabel("Growth")

plt.legend()

plt.show()


# Probability Distribution

plt.figure(figsize=(10, 5))

plt.hist(df["Probability_Up"], bins=50)

plt.title("Prediction Probability Distribution")

plt.xlabel("Probability of UP")
plt.ylabel("Frequency")

plt.show()


# Trade Signals Visualization

plt.figure(figsize=(12,4))

plt.plot(df["Strategy_Signal"])

plt.title("Trading Signals Over Time")

plt.xlabel("Time")
plt.ylabel("Signal")

plt.show()

# Drawdown Calculations: important risk metrics in finance
rolling_max = (df["Cumulative_Strategy"].cummax())

draw_down = (df["Cumulative_Strategy"] - rolling_max) / rolling_max

df["DrawDown"] = draw_down

#print(df["DrawDown"].head())

plt.figure(figsize=(12,5))

plt.plot(df["DrawDown"])

plt.title("Strategy Drawdown")

plt.xlabel("Time")
plt.ylabel("DrawDown")

plt.show()

# Basic Sharpe Ratio: return per unit of risk

sharpe = (
    df["Strategy_Return"].mean() / df["Strategy_Return"].std()
) * np.sqrt(252)

print("\nSharpe Ratio")
print("----------------")

print(round(sharpe, 4))


df.to_csv("results/visualized_backtest.csv", index=False)
