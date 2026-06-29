import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
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

standard_eval_results = []

threshold = 0.5

probabilities = model.predict_proba(X_test_scaled)[:, 1]

predictions = (probabilities >= threshold).astype(int)

accuracy =  accuracy_score(y_test, predictions)

report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

bullish_rate = (predictions.mean()) * 100

prob_summary_results = pd.Series(probabilities).describe()

confidence = np.maximum(probabilities, 1-probabilities)
confidence_stats_results = pd.Series(confidence).describe()

prediciton_distribution = pd.Series(predictions).value_counts().sort_index()

bullish_preds = (predictions == 1).sum()
bearish_preds = (predictions == 0).sum()

bullish_pctg = bullish_preds / len(predictions) * 100
bearish_pctg = bearish_preds / len(predictions) * 100

importance = pd.DataFrame({
    "Feature":features,
    "Importance":model.feature_importances_
})

importance = importance.sort_values(ascending=False, by='Importance')

## RESEARCH SUMMARY ##
print("\nRESEARCH SUMMARY")
print("---------------------------")

print("Best Sample Weight : 1.09")
print("Best Threshold     : 0.50")
print("Final Accuracy     :", round(accuracy,4))

print("\nMajor Findings")
print("---------------------------")
print("- Probability calibration showed the model has very low confidence.")
print("- Threshold tuning did not improve overall performance.")
print("- Sample weighting reduced bullish bias but only slightly.")
print("- Error analysis showed correct and incorrect predictions have very similar feature values.")
print("- Current feature set has reached its predictive limit.")

print("\nNext Phase")
print("---------------------------")
print("- AWS Deployment")
print("- Automated Daily Prediction Pipeline")
print("- Rich Feature Engineering")

## Benchmark Table ##

benchmark = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "F1 Score",

        "Bullish Recall",

        "Bearish Recall",

        "Average Probability",

        "Average Confidence",

        "Bullish Prediction %",

        "False Positives",

        "False Negatives",

        "Total Features"

    ],

    "Value":[

        round(accuracy),

        round(f1_score(y_test,predictions),4),

        round(report["1"]["recall"],4),

        round(report["0"]["recall"],4),

        round(probabilities.mean(),4),

        round(confidence.mean(),4),

        round(bullish_rate,2),

        confusion_matrix(y_test,predictions)[0][1],

        confusion_matrix(y_test,predictions)[1][0],

        len(features)

    ]

})

print("\nFINAL BENCHMARK")
print("---------------------------")
print(benchmark)

## PROBABILITY SUMMARY ##
print("\nPROBABILITY SUMMARY")
print("-------------------------")
print(prob_summary_results)

## CONFIDENCE SUMMARY ##
print("\nCONFIDENCE SUMMARY")
print("------------------------")
print(confidence_stats_results)

## Final Evaluation ##
print("\nMODEL ACCURACY")
print("---------------------------")
print(round(accuracy,4))

print("\nCONFUSION MATRIX")
print("---------------------------")
print(confusion_matrix(y_test,predictions))

print("\nCLASSIFICATION REPORT")
print("---------------------------")
print(classification_report(y_test,predictions,zero_division=0))

## FINAL FEATURE IMPORTANCE ##
print("\nFINAL FEATURE IMPORTANCE")
print("------------------------")
print(importance)

## PREDICTION SUMMARY ##
print("\nPREDICTION DISTRIBUTION")
print("---------------------------")
print(prediciton_distribution)

print("\nBullish Predictions :", bullish_preds)
print("Bearish Predictions :", bearish_preds)
print("Bullish % :", round(bullish_pctg,2))
print("Bearish % :", round(bearish_pctg,2))

## SAVING THE WORK ##

benchmark.to_csv("reports/week6_benchmark.csv", index=False)
print("Benchmark saved successfully.")

importance.to_csv("reports/week6_feature_importance.csv", index=False)
print("Feature Importance saved successfully.")

with open(
    "reports/week6_summary.txt",
    "w"
) as file:

    file.write("WEEK 6 FINAL BENCHMARK\n")
    file.write("="*40+"\n\n")

    file.write(f"Accuracy : {accuracy:.4f}\n")
    file.write(f"F1 Score : {f1_score(y_test,predictions):.4f}\n")
    file.write(f"Bullish Recall : {report['1']['recall']:.4f}\n")
    file.write(f"Bearish Recall : {report['0']['recall']:.4f}\n")

    file.write("\n")

    file.write(f"Average Probability : {probabilities.mean():.4f}\n")
    file.write(f"Average Confidence : {confidence.mean():.4f}\n")

    file.write("\n")

    file.write("Key Findings\n")
    file.write("-------------------------\n")

    file.write(
        "- Sample weighting improved class balance.\n"
    )

    file.write(
        "- Threshold tuning did not significantly improve performance.\n"
    )

    file.write(
        "- Correct and incorrect predictions have nearly identical feature distributions.\n"
    )

    file.write(
        "- The current feature set has reached its predictive limit.\n"
    )

    file.write(
        "- Future work will focus on richer features and AWS deployment.\n"
    )

print("Summary report saved.")