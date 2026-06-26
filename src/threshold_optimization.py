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

probabilities = model.predict_proba(X_test_scaled)[:,1]

thresholds = np.arange(0.45, 0.61, 0.01)

results = []

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    bullish_rate = (predictions.mean()) * 100

    results.append({
        "Threshold": round(threshold, 2),
        "Acccuracy": round(accuracy, 2),
        "Bullish Recall": round(report['1']['recall'], 4),
        "Bearish Recall": round(report['0']['recall'], 4),
        "Bullish Prediction %": round(bullish_rate, 2)
    })

results_df = pd.DataFrame(results)
print(results_df)

