import pandas as pd
import numpy as np
import os
import json
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import(
    classification_report,
    f1_score,
    confusion_matrix,
    accuracy_score
)

from xgboost import XGBClassifier

artifact_dir = "artifacts"

os.makedirs(f"{artifact_dir}/models", exist_ok=True)
os.makedirs(f"{artifact_dir}/metrics", exist_ok=True)
os.makedirs(f"{artifact_dir}/features", exist_ok=True)

FEATURES = [

    "Daily_Return",

    "Return_Lag1",

    "Return_Lag2",

    "Return_Lag3",

    "Volatility_Ratio",

    "Volatility_Trend",

    "Return_Volume",

    "VolRation_Volume",

]

TARGET = "Target"

def train_model(df, sample_weight_value=1.09):

    X = df[FEATURES]

    y = df[TARGET]

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sample_weights = y_train.apply(lambda x: sample_weight_value if x == 0 else 1.0)

    model = XGBClassifier(
        n_estimators = 50,
        max_depth = 2,
        random_state = 42,
        learning_rate = 0.01
    )

    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    metrics = {

        "accuracy":

            accuracy_score(
                y_test,
                predictions,
            ),

        "classification_report":

            classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0,
            ),

        "confusion_matrix":

            confusion_matrix(
                y_test,
                predictions,
            ),

        "f1_score":

            f1_score(
                y_test,
                predictions,
            ),

        "probabilities":

            probabilities,

        "predictions":

            predictions,

        "actual":

            y_test.values,

        "scaler":

            scaler,

    }

    metrics_to_save = {
    "accuracy": float(metrics["accuracy"]),
    "f1_score": float(metrics["f1_score"]),
    "classification_report": metrics["classification_report"],
    "confusion_matrix": metrics["confusion_matrix"].tolist()
    }

    model.save_model(f"{artifact_dir}/models/xgboost_financial_model_v1.json")

    joblib.dump(scaler, f"{artifact_dir}/models/standard_scaler.pkl")

    with open(f"{artifact_dir}/metrics/training_metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=4)
    
    with open(f"{artifact_dir}/features/feature_names.json", "w") as f:
        json.dump(FEATURES, f, indent=4)

    return scaler, model, metrics
