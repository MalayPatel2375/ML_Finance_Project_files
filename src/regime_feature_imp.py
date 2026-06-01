import pandas as pd

from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/model_input/model_ready_data.csv")

df = df.drop(columns=["Unnamed: 0"])

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

df["Market_Regime"] = df["Daily_Return"].apply(
    lambda x: "Bullish" if x >= 0 else "Bearish"
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

    "Price_VS_SMA20"
]

for regime in ["Bullish", "Bearish"]:
    regime_df = df[df["Market_Regime"] == regime]

    if len(regime_df) < 1000:
        continue

    X = regime_df[selected_features]
    y = regime_df["Target"]

    split_index = int(len(regime_df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators = 50,
        random_state = 42,
        max_depth = 2,
        learning_rate = 0.01
    )

    model.fit(X_train_scaled, y_train)

    importance_df = pd.DataFrame({
        "Feature": selected_features,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    print(f"\n{regime.upper()} REGIME")
    print("--------------------")

    print(importance_df)

    importance_df.to_csv(f"results/{regime}_feature_importance.csv", index=False)

    