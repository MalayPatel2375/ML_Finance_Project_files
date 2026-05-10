import pandas as pd

df = pd.read_csv("data/processed/validated_data.csv")

print("==========DATA QUALITY REPORT===========")

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nMissing Values: ")
print(df.isnull().sum())

print("\nDuplicate Rows: ")
print(df.duplicated().sum())

print("\nUnique Tickers: ")
print(df['Ticker'].nunique())

print("\nTicker Counts: ")
print(df['Ticker'].value_counts())