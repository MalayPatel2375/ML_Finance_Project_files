import pandas as pd

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

bearish_weights = [1, 1.04, 1.05, 1.06, 1.07, 1.08,1.09, 1.1]

results = []

for weight in bearish_weights:

    sample_weights = y_train.apply(lambda x: weight if x==0 else 1.0)
    
    model = XGBClassifier(
        n_estimators = 50,
        max_depth = 2,
        learning_rate = 0.01,
        random_state = 42
    )

    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(y_test, predictions, output_dict=True)
    print(report)

    bullish_recall = report['1']['recall']
    bearish_recall = report['0']['recall']

    bullish_predictions = (predictions.sum() / len(predictions))*100

    results.append({
        "Weight": weight,
        "Accuracy": round(accuracy, 4),
        "Bullish_Recalls": round(bullish_recall, 4),
        "Bearish_Recalls": round(bearish_recall, 4),
        "Bullish_Predictions_%": round(bullish_predictions, 2)
    })

results_df = pd.DataFrame(results)

print("CLASS WEIGHT COMPARISON")
print("-----------------------")
print(results_df)

results_df.to_csv("results/class_weight_optimization_result.csv", index=False)