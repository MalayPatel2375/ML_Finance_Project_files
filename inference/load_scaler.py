import joblib 
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

SCALER_PATH = Path("artifacts/models/standard_scaler.pkl")

def load_scaler():
    scaler = joblib.load(SCALER_PATH)
    logger.info("Scaler Loaded Successfully.")
    return scaler
