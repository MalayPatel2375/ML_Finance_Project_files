from inference.load_model import load_xgboost_model
from inference.load_scaler import load_scaler
from inference.predict import predict_single
from fastapi import HTTPException
import pandas as pd
import numpy as np

model = load_xgboost_model()
scaler = load_scaler()

def make_prediction(features: dict):

    try:
        sample = pd.DataFrame([features])

        result = predict_single(
            model=model,
            scaler=scaler,
            sample=sample
        )

        result["Model"] = "Financial XGBoost"
        result["Version"] = "V1"

        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))