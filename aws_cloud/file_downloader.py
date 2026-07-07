import boto3
from pathlib import Path

BUCKET = "malay-ml-sagemaker"

s3 = boto3.client("s3")

def download_file(s3_key, local_path):
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(BUCKET, s3_key, str(local_path))

    print(f"Downloaded: {local_path}")
