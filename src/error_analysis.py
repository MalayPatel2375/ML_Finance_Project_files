import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from xgboost import XGBClassifier

features = [

    "Daily_Return",

    "Return_Lag1",

    "Return_Lag2",

    "Return_Lag3",

    "Volatility_Ratio",

    "Volatility_Trend",

    "Return_Volume",

    "VolRation_Volume"
]

df = pd.read_csv("data/model_input/model_interaction_data.csv")

target = "Target"

X = df[features]
y = df[target]

split_index = int(len(df)*0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

bearish_weights = 1.09 ##best weight from Day4 for the model

sample_weights = y_train.apply(
    lambda x: bearish_weights if x==0 else 1.0
)

model = XGBClassifier(
    n_estimators = 50,
    max_depth = 2,
    learning_rate = 0.01,
    random_state = 42
)

model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

threshold = 0.5

probabilties = model.predict_proba(X_test_scaled)[:,1]
predictions = (probabilties >= 0.5).astype(int)

analysis = X_test.copy()

analysis["Actual"] = y_test.values
analysis["Predictions"] = predictions
analysis["Probabilities"] = probabilties

analysis["Result"] = "Correct"
analysis.loc[analysis["Actual"] != analysis["Predictions"], "Result"] = "Incorrect"

false_positive = analysis[(analysis["Actual"]==0) & (analysis["Predictions"]==1)]
false_negative = analysis[(analysis["Actual"]==1) & (analysis["Predictions"]==0)]

print("TOTAL TEST SAMPLES")
print(len(analysis))

print()

print("CORRECT")
print((analysis["Result"]=="Correct").sum())

print()

print("INCORRECT")
print((analysis["Result"]=="Incorrect").sum())

print()

print("FALSE POSITIVE")
print(len(false_positive))

print()

print("FALSE NEGATIVE")
print(len(false_negative))

correct = analysis[analysis["Result"] == "Correct"]
incorrect = analysis[analysis["Result"] == "Incorrect"]

comparison = pd.DataFrame({
    "Correct": correct.mean(numeric_only=True),
    "Incorrect": incorrect.mean(numeric_only=True)
})

print(comparison)

print("Average Confidence (Correct)")
print(correct["Probabilities"].apply(lambda x:max(x, 1-x)).mean())

print("Average Confidence (Incorrect)")
print(incorrect["Probabilities"].apply(lambda x:max(x, 1-x)).mean())

incorrect["Confidence"] = incorrect["Probabilities"].apply(lambda x:max(x, 1-x))

print(incorrect.sort_values("Confidence", ascending=False).head(20))

