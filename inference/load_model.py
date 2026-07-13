from pathlib import Path
from xgboost import XGBClassifier
from utils.logger import get_logger

MODEL_PATH = Path("artifacts/models/xgboost_financial_model_v1.json")

logger = get_logger(__name__)

def load_xgboost_model():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    logger.info("Model Loaded Successfully.")
    return model
