from pathlib import Path
from xgboost import XGBClassifier

MODEL_PATH = Path("artifacts/models/xgboost_financial_model_v1.json")

def load_xgboost_model():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    print("Model Loaded Successfully.")
    return model
