import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/validated_data.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(by=["Ticker", "Date"])

#print(df.head())

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

#print(df[["Ticker", "Date", "Close", "Daily_Return", "Target"]].head(10))

#print(df.isnull().sum())

df = df.dropna()

df.to_csv("data/features/basic_features.csv", index=False)

#print(df.shape)

#print(df["Target"].value_counts(normalize=True))

#Simple Moving Avgs
df["SMA_5"] = (
    df.groupby("Ticker")["Close"].transform(
        lambda x: x.rolling(window=5).mean()
    )
)

df["SMA_10"] = (
    df.groupby("Ticker")["Close"].transform(
        lambda x: x.rolling(window=10).mean()
    )
)

df["SMA_20"] = (
    df.groupby("Ticker")["Close"].transform(
        lambda x: x.rolling(window=20).mean()
    )
)

#Volatiltiy
df["Volatility_5"] = (
    df.groupby("Ticker")["Daily_Return"].transform( #using daily_return to measure movement magnitude
        lambda x: x.rolling(window=5).std()
    )
)

#Price Distance features: measures how far price is from trend
df["Price_VS_SMA5"] = (
    (df["Close"] - df["SMA_5"]) / df["SMA_5"]
)

df["Price_VS_SMA20"] = (
    (df["Close"] - df["SMA_20"]) / df["SMA_20"]
)

#print(df[["Ticker", "Date", "Close", "SMA_5", "SMA_10", "SMA_20", "Volatility_5"]].head(15))

#print(df.isnull().sum())

df = df.dropna()

df.to_csv("data/features/technical_features.csv", index=False)

print(df.shape)

print(df.columns)