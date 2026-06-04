import pandas as pd
import numpy as np

df = pd.read_csv("data/model_input/model_volume_data.csv")

df = df.drop(columns=[
    "Breakout_5",
    "Breakout_10",
    "Breakdown_5",
    "Breakdown_10",
    "High_10",
    "High_20",
    "Distance_From_High",
    "RVOL",
    "Volume_VS_Avg",
    "Volume_Momentum"
])

df["Volatility_20"] = (
    df["Daily_Return"].rolling(20).std()
)

df["Volatility_Ratio"] = (
    df["Volatility_5"] / df["Volatility_20"]
)

df["Volatility_Expansion"] = np.where(df["Volatility_Ratio"] > 1.2, 1, 0)

df["Volatility_Contraction"] = np.where(df["Volatility_Ratio"] < 0.8, 1, 0)

df["Volatility_Trend"] = (df["Volatility_5"] - df["Volatility_5"].shift(5))

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

df.to_csv("data/model_input/model_volatility_data.csv", index=False)

# QUICK VALIDATION 

new_features = [

    "Volatility_Ratio",

    "Volatility_Expansion",

    "Volatility_Contraction",

    "Volatility_Trend"
]

print("\nFEATURE SUMMARY")
print("--------------------")

print(
    df[new_features]
    .describe()
)







# TESTING MODEL WITH NEW FEATURES

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv(
    "data/model_input/model_volatility_data.csv"
)

selected_features = [

    "Daily_Return",

    "RSI_14",

    "Momentum_5",

    "Volatility_5",

    "Return_Lag1",

    "Return_Lag2",

    "Return_Lag3",

    "Price_VS_SMA5",

    "Price_VS_SMA20",

    "Distance_From_Low",

    "Volume_Change",

    "Volatility_Ratio",

    "Volatility_Expansion",

    "Volatility_Contraction",

    "Volatility_Trend"
]

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