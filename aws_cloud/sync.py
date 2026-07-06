from pathlib import Path
from aws_cloud.config import (
    RAW_DIR,
    PROCESSED_DIR,
    FEATURE_DIR,
    MODEL_INPUT_DIR,
    RAW_PREFIX,
    PROCESSED_PREFIX,
    FEATURE_PREFIX,
    MODEL_INPUT_PREFIX
)
from aws_cloud.s3_utils import file_hash
from aws_cloud.manifest import save_manifest, load_manifest

from aws_cloud.upload import upload_file

FOLDERS = [(RAW_DIR, RAW_PREFIX),
           (PROCESSED_DIR, PROCESSED_PREFIX),
           (FEATURE_DIR, FEATURE_PREFIX),
           (MODEL_INPUT_DIR, MODEL_INPUT_PREFIX)]

manifest = load_manifest()

uploaded = 0
skipped = 0

print("\nScanning Project......\n")

for local_folder, prefix in FOLDERS:
    for file in Path(local_folder).rglob("*"):
        if not file.is_file():
            continue

        relative_path = file.relative_to(local_folder)
        s3_key = prefix + str(relative_path).replace("\\", "/")
        md5 = file_hash(file)

        if manifest.get(s3_key) == md5:
            skipped += 1
            print(f"Skipped: {s3_key}")
            continue

        upload_file(file, s3_key)
        manifest[s3_key] = md5
        uploaded += 1
    
save_manifest(manifest)

print("\n==========================")
print("Synchronization Complete")
print("==========================")
print(f"Uploaded : {uploaded}")
print(f"Skipped  : {skipped}")