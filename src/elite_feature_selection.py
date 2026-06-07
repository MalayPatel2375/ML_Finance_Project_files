import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("data/model_input/model_interaction_data.csv")

elite_features = [

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

X = df[elite_features]
y = df["Target"]

split_index = int(len(df) * 0.8)
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBClassifier(
    n_estimators = 50,
    max_depth = 2,
    learning_rate = 0.01,
    random_state = 42
)

model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)

print("ELITE FEATURE MODEL RESULTS")
print("---------------------")
print(f"\nAccuracy: {round(accuracy, 4)}")

print("------------------------")
print(f"\nConfusion Matrix: {confusion_matrix(y_test, predictions)}")

print("------------------------")
print(f"\nClassification Report: {classification_report(y_test, predictions)}")

importance_df = pd.DataFrame({
    "Features": elite_features,
    "Importance": model.feature_importances_
})

importance_df = (importance_df.sort_values(by="Importance", ascending=False))

print("\nELITE FEATURE IMPORTANCE")
print("----------------------------")

print(importance_df)

importance_df.to_csv("results/elite_feature_importance.csv", index=False)



#        FEATURE STABILITY TEST

results = []

features_to_test = [

    "RSI_14",

    "Distance_From_Low",

    "Volatility_Trend",

    "Return_Volume",

    "VolRation_Volume"

]

for feature in features_to_test:

    test_features = elite_features.copy()

    test_features.remove(feature)

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

        "Removed_Feature": feature,

        "Accuracy": round(
            accuracy,
            4
        )
    })

results_df = pd.DataFrame(
    results
)

print("\nFEATURE STABILITY RESULTS")
print("-------------------------")

print(
    results_df.sort_values(
        by="Accuracy",
        ascending=False
    )
)