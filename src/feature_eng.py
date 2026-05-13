import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/validated_data.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(by=["Ticker", "Date"])

print(df.head())

# Daily returns
df["Daily_Return"] = (
    df.groupby("Ticker")["Close"].pct_change()
)

# Target variable with supervised learning
# comparing tomorrow's price with today's price by shifting to -1
df["Target"] = np.where(
    df.groupby("Ticker")["Close"].shift(-1) > df["Close"],
    1,
    0
)

print(df[["Ticker", "Date", "Close", "Daily_Return", "Target"]].head(10))

print(df.isnull().sum())

df = df.dropna()

df.to_csv("data/features/basic_features.csv", index=False)

print(df.shape)

print(df["Target"].value_counts(normalize=True))