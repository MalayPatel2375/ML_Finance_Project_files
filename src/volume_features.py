import pandas as pd
import numpy as np

df = pd.read_csv("data/model_input/model_structure_data.csv")

df["Volume_MA20"] = (
    df["Volume"].rolling(20).mean()
)

df["RVOL"] = (df["Volume"] / df["Volume_MA20"])

df["Volume_Change"] = df["Volume"].pct_change()

df["Volume_Momentum"] = (df["Volume"] / df["Volume"].shift(5))

df["Volume_VS_Avg"] = (
    (df["Volume"] - df["Volume_MA20"]) / df["Volume_MA20"]
)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

df.to_csv("data/model_input/model_volume_data.csv", index=False)

# Quick Validation

new_features = ["RVOL", "Volume_Change", "Volume_Momentum", "Volume_VS_Avg"]

print("\nFeature Summary")
print("-------------------")
print(df[new_features].describe())











# MODEL TESTING WITH NEW FEATURES

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

# Load data
df = pd.read_csv(
    "data/model_input/model_volume_data.csv"
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

    "Distance_From_Low"
    
]

volume_structure_features = [

    "RVOL",
    "Volume_Change",
    "Volume_Momentum",
    "Volume_VS_Avg"

]

# Combined feature set
selected_features = (
    original_features +
    market_structure_features +
    volume_structure_features
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