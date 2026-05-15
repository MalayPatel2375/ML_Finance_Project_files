import pandas as pd
import numpy as np

df = pd.read_csv("data/features/enhanced_features.csv")

df["Date"] = pd.to_datetime(df["Date"])

#print(df.shape)
#print(df.head())

#print("\nMissing Values")
#print(df.isnull().sum())

df = df.dropna()

#print(df.isnull().sum())
#print(df.shape)

duplicates = df.duplicated().sum()
#print(f"Duplicate Rows: {duplicates}")

constant_cols = [
    col for col in df.columns
    if df[col].nunique() <= 1
]

#print(f"\nConstant Columns: {constant_cols}")

numeric_df = df.select_dtypes(include=np.number)
correlation_matrix = numeric_df.corr()

#print(correlation_matrix["Target"].sort_values(ascending=False))

future_cols = [
    col for col in df.columns
    if "Target" in col and col != "Target"
]

#print(f"\nPotential Leakage Columns: {future_cols}")

#print(f"\nFeature Statistics: {numeric_df.describe()}")

#print(f"\nTarget Distribution: {df["Target"].value_counts(normalize=True)}")

model_df = df.drop(columns=["Date", "Ticker"])

model_df.to_csv("data/model_input/model_ready_data.csv", index=False)
#print("\nModel-ready dataset saved successfully.")

print(model_df.shape)
print(model_df.columns)