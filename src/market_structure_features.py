import pandas as pd
import numpy as np

df = pd.read_csv("data/model_input/model_ready_data.csv")

df = df.drop(columns=["Unnamed: 0"])

df["High_5"] = (df["High"].rolling(5).max())

df["High_10"] = (df["High"].rolling(10).max())

df["Low_5"] = (df["Low"].rolling(5).min())

df["Low_10"] = (df["Low"].rolling(10).min())

df["Breakout_5"] = np.where(df["Close"] > df["High_5"].shift(1), 1, 0)

df["Breakout_10"] = np.where(df["Close"] > df["High_10"].shift(1), 1, 0)

df["Breakdown_5"] = np.where(df["Close"] < df["Low_5"].shift(1), 1, 0)

df["Breakdown_10"] = np.where(df["Close"] < df["Low_10"].shift(1), 1, 0)

df["High_20"] = (df["High"].rolling(20).max())

df["Low_20"] = (df["Low"].rolling(20).min())

df["Distance_From_High"] = ((df["Close"] - df["High_20"]) / df["High_20"])

df["Distance_From_Low"] = ((df["Close"] - df["Low_20"]) / df["Low_20"])

print("\nMissing Values")
print("-----------------------")

print(df[["Breakout_5", "Breakout_10", "Breakdown_5", "Breakdown_10", "Distance_From_High", "Distance_From_Low"]].isna().sum())

df = df.dropna()

df.to_csv("data/model_input/model_structure_data.csv", index=False)

print("\nFeature Counts")
print("-------------------")

print(df[["Breakout_5", "Breakout_10", "Breakdown_5", "Breakdown_10"]].sum())

print("\nNew Feature Summary")
print("-------------------")

print(df[["Distance_From_High", "Distance_From_Low"]].describe())


# Quick Check with new features

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

# Load data
df = pd.read_csv(
    "data/model_input/model_structure_data.csv"
)

# Original features
original_features = [

    "Daily_Return",
    "RSI_14",
    "Momentum_5",
    "Volatility_5",
    "Return_Lag1",
    "Return_Lag2",
    "Return_Lag3",
    "Price_VS_SMA5",
    "Price_VS_SMA20"
]

# New market structure features
market_structure_features = [

    "Breakout_5",
    "Breakout_10",
    "Breakdown_5",
    "Breakdown_10",
    "Distance_From_High",
    "Distance_From_Low"
]

# Combined feature set
selected_features = (
    original_features +
    market_structure_features
)

X = df[selected_features]
y = df["Target"]

# Time split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Scale
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# Model
model = XGBClassifier(
    n_estimators=50,
    max_depth=2,
    learning_rate=0.01,
    random_state=42
)

# Train
model.fit(
    X_train_scaled,
    y_train
)

# Predict
predictions = model.predict(
    X_test_scaled
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nMODEL ACCURACY")
print("----------------")
print(round(accuracy, 4))

# Feature Importance
importance_df = pd.DataFrame({

    "Feature": selected_features,

    "Importance":
    model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 15 FEATURES")
print("----------------")
print(
    importance_df.head(15)
)