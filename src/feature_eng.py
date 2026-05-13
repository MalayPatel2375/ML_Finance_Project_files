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

#print(df.shape)

#print(df.columns)

#Momentum: measures how strongly price moved recently
df["Momentum_5"] = (
    df.groupby("Ticker")["Close"].transform(
        lambda x: x - x.shift(5)
    )
)

#RSI: Relative Strength Index

delta = df.groupby("Ticker")["Close"].diff()

gain = delta.clip(lower=0)

loss = -delta.clip(upper=0)

avg_gain = gain.groupby(df["Ticker"]).transform(
    lambda x: x.rolling(window=14).mean()
)

avg_loss = loss.groupby(df["Ticker"]).transform(
    lambda x: x.rolling(window=14).mean()
)

rs = avg_gain / avg_loss

df["RSI_14"] = 100 - (100 / (1 + rs))

# RSI Interpretation: 
    # > 70: potentially overbrought
    # < 30: potentially oversold

# Lagged Returns

df["Return_Lag1"] = (
    df.groupby("Ticker")["Daily_Return"]
    .shift(1)
)

df["Return_Lag2"] = (
    df.groupby("Ticker")["Daily_Return"]
    .shift(2)
)

df["Return_Lag3"] = (
    df.groupby("Ticker")["Daily_Return"]
    .shift(3)
)

# lag feature matters, they provide memory to model: here upto 3 days ago

# Lagged Prices

df["Close_Lag1"] = (
    df.groupby("Ticker")["Close"]
    .shift(1)
)

df["Close_Lag2"] = (
    df.groupby("Ticker")["Close"]
    .shift(2)
)

# why it matters: provides price memory

#print(df[["Ticker", "Date", "Close", "Momentum_5", "RSI_14", "Return_Lag1", "Close_Lag2"]].head(20))

#print(df.isnull().sum())

#df = df.dropna()

df.to_csv("data/features/enhanced_features.csv", index=False)

#print(df.shape)

#print(df.columns)

