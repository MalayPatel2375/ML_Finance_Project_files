import yfinance as yf
import pandas as pd
import os
from config.settings import TICKERS, START_DATE, END_DATE

OUTPUT_PATH = "data/raw/raw_data.csv"

def fetch_data():
    all_data = []

    for ticker in TICKERS:
        print(f"Fetching data for {ticker}...")

        df = yf.download(ticker, start=START_DATE, end=END_DATE)

        if df.empty:
            print(f"Warning: No data for {ticker}")
            continue

        df["Ticker"] = ticker
        df.reset_index(inplace=True)

        all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)
    return final_df

def save_data(df):
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    data = fetch_data()
    save_data(data)