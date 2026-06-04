import pandas as pd
import numpy as np

df = pd.read_csv("data/model_input/model_volatility_data.csv")

df["RSI_Volume"] = (df["RSI_14"] * df["Volume_Change"])

df["Return_Volume"] = df["Daily_Return"] * df["Volume_Change"]

df["VolRation_Volume"] = df["Volatility_Ratio"] * df["Volume_Change"]

df["RSI_VolRation"] = df["RSI_14"] * df["Volatility_Ratio"]

df["LowDist_Volume"] = df["Distance_From_Low"] * df["Volume_Change"]

df["Return_VolRation"] = df["Daily_Return"] * df["Volatility_Ratio"]

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

df.to_csv("data/model_input/model_interaction_data.csv", index=False)

# QUICK VALIDATION

interaction_features = [

    "RSI_Volume",

    "Return_Volume",

    "VolRation_Volume",

    "RSI_VolRation",

    "LowDist_Volume",

    "Return_VolRation"
]

print("\nINTERACTION FEATURE SUMMARY")
print("----------------------------")

print(
    df[interaction_features]
    .describe()
)






# MODEL TESTING WITH NEW FEATURES

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv(
    "data/model_input/model_interaction_data.csv"
)

selected_features = [

    "Daily_Return",

    "RSI_14",

    "Return_Lag1",

    "Return_Lag2",

    "Return_Lag3",

    "Price_VS_SMA20",

    "Distance_From_Low",

    "Volume_Change",

    "Volatility_Ratio",

    "Volatility_Trend",

    "RSI_Volume",

    "Return_Volume",

    "VolRation_Volume",

    "RSI_VolRation",

    "LowDist_Volume",

    "Return_VolRation"
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