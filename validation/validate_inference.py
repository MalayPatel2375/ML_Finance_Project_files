import json
import pandas as pd
import numpy as np
import boto3

from inference.predict import predict, predict_single
from inference.load_model import load_xgboost_model
from inference.load_scaler import load_scaler

def validate():

    print("=" * 60)
    print("Starting Inference Validation")
    print("=" * 60)

    #Load artifacts
    model = load_xgboost_model()
    scaler = load_scaler()

    print("✓ Model Loaded")
    print("✓ Scaler Loaded")

    #Load feature names
    with open("artifacts/features/feature_names.json") as f:
        features = json.load(f)
    
    print(f"✓ Loaded {len(features)} features")

    #Load dataset from S3
    bucket = "malay-ml-sagemaker"
    key = "data/model_input/model_interaction_data.csv"

    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)

    df = pd.read_csv(obj["Body"])

    X = df[features]

    #Batch predictions
    batch = predict(model, scaler, X.head(10))
    print(f"✓ Batch Prediction Successful ({len(batch)} rows)")

    #Single prediction
    single = predict_single(model, scaler, X.iloc[0])
    print("✓ Single Prediction Successful")

    print(single)

    print("=" * 60)
    print("Inference Validation PASSED")
    print("=" * 60)

if __name__ == "__main__":
    validate()
