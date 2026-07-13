import boto3
from pathlib import Path
from config.aws_config import BUCKET_NAME

BUCKET = BUCKET_NAME

s3 = boto3.client("s3")

def download_file(s3_key, local_path):
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(BUCKET, s3_key, str(local_path))

    print(f"Downloaded: {local_path}")
