from pathlib import Path
from config.aws_config import BUCKET_NAME, AWS_REGION

# AWS 
AWS_REGION = AWS_REGION
BUCKET_NAME = BUCKET_NAME

# PROJECT ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# LOCAL DIRECTORIES
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"

PROCESSED_DIR = DATA_DIR / "processed"

FEATURE_DIR = DATA_DIR / "features"

MODEL_INPUT_DIR = DATA_DIR / "model_input"

PREDICTION_DIR = DATA_DIR / "predictions"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"

LOGS_DIR = PROJECT_ROOT / "logs"

# S3 PREFIXES
RAW_PREFIX = "data/raw/"

PROCESSED_PREFIX = "data/processed/"

FEATURE_PREFIX = "data/features/"

MODEL_INPUT_PREFIX = "data/model_input/"

PREDICTION_PREFIX = "data/predictions/"

MODEL_PREFIX = "models/xgboost/"