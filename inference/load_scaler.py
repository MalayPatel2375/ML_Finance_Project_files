import joblib 
from pathlib import Path

SCALER_PATH = Path("artifacts/models/standard_scaler.pkl")

def load_scaler():
    scaler = joblib.load(SCALER_PATH)
    print("Scaler Loaded Successfully.")
    return scaler
