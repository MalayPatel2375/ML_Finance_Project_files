from dotenv import load_dotenv
import os

"""
Project Configuration
"""

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")

S3_BUCKET = os.getenv("S3_BUCKET")

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

MODEL_NAME = "Financial XGBoost"

AUTHOR = "Malay Patel"

FRAMEWORK = "XGBoost"

PROJECT_NAME = "Financial Machine Learning Project"