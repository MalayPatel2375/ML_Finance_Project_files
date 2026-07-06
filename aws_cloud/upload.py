from aws_cloud.s3_client import get_s3_client
from aws_cloud.config import BUCKET_NAME

s3 = get_s3_client()

def upload_file(local_path, s3_key):

    s3.upload_file(str(local_path), BUCKET_NAME, s3_key)

    print(f"Uploaded: {s3_key}")

