from pathlib import Path
from aws_cloud.file_downloader import download_file

files = {

    "models/xgboost/xgboost_financial_model_v1.json":

    Path("artifacts/models/xgboost_financial_model_v1.json"),

    "models/xgboost/training_metrics.json":

    Path("artifacts/metrics/training_metrics.json"),

    "models/xgboost/feature_names.json":

    Path("artifacts/features/feature_names.json"),

}

for s3_key, local_path in files.items():
    download_file(s3_key, local_path)