"""
Artifact Configuration
"""

from pathlib import Path

ARTIFACT_ROOT = Path("artifacts")

MODEL_DIR = ARTIFACT_ROOT / "models"

METRICS_DIR = ARTIFACT_ROOT / "metrics"

FEATURE_DIR = ARTIFACT_ROOT / "features"

MODEL_PATH = MODEL_DIR / "xgboost_financial_model_v1.json"

SCALER_PATH = MODEL_DIR / "standard_scaler.pkl"

METRICS_PATH = METRICS_DIR / "training_metrics.json"

FEATURE_PATH = FEATURE_DIR / "feature_names.json"