import yfinance as yf
import pandas as pd
from config.settings import TICKERS, START_DATE, END_DATE
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

OUTPUT_PATH = "data/raw/raw_data.csv"

def fetch_data():
    all_data = []

    for ticker in TICKERS:
        logging.info(f"Fetching {ticker}")

        try:
            df = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=False
            )

            print(df.columns)

            if df.empty:
                logging.warning(f"No data for {ticker}")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.reset_index(inplace=True)

            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

            df["Ticker"] = ticker

            all_data.append(df)

        except Exception as e:
            logging.error(f"{ticker}: {e}")

    final_df = pd.concat(all_data, ignore_index=True)

    return final_df


def save_data(df):
    os.makedirs("data/raw", exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"[INFO] Data saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    data = fetch_data()
    save_data(data)