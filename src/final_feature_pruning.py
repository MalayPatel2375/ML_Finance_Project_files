import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

df = pd.read_csv("data/model_input/model_interaction_data.csv")

base_features = [

    "Daily_Return",

    "RSI_14",

    "Return_Lag1",

    "Return_Lag2",

    "Return_Lag3",

    "Price_VS_SMA20",

    "Distance_From_Low",

    "Volatility_Ratio",

    "Volatility_Trend",

    "Return_Volume",

    "VolRation_Volume"
]

baseline_accuracy = 0.526
print(f"BASELINE ACCURACY: {baseline_accuracy}")

features_to_remove = [

    "RSI_14",

    "Distance_From_Low",

    "Price_VS_SMA20"
]

results = []


for feature in features_to_remove:

    test_features = base_features.copy()

    test_features.remove(
        feature
    )

    X = df[test_features]

    y = df["Target"]

    split_index = int(len(df) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = XGBClassifier(

        n_estimators=50,

        max_depth=2,

        learning_rate=0.01,

        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results.append({

        "Removed_Feature":
        feature,

        "Accuracy":
        round(
            accuracy,
            4
        ),

        "Accuracy_Change":
        round(
            accuracy -
            baseline_accuracy,
            4
        )
    })


# ==========================
# RESULTS TABLE
# ==========================

results_df = pd.DataFrame(
    results
)

print("\nPRUNING RESULTS")
print("---------------------")

print(
    results_df.sort_values(
        by="Accuracy",
        ascending=False
    )
)

results_df.to_csv("results/elite_features_acc_change.csv", index=False)

# ==========================
# FINAL DECISION
# ==========================

print("\nFEATURE DECISIONS")
print("---------------------")

for _, row in results_df.iterrows():

    if abs(row["Accuracy_Change"]) < 0.001:

        decision = "REMOVE"

    else:

        decision = "KEEP"

    print(

        f"{row['Removed_Feature']} "
        f"-> {decision}"
    )

#########################
# FINAL FEATURE SET
#########################

final_features = [

    "Daily_Return",

    "Return_Lag1",

    "Return_Lag2",

    "Return_Lag3",

    "Volatility_Ratio",

    "Volatility_Trend",

    "Return_Volume",

    "VolRation_Volume"
]