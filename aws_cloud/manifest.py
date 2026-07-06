import json
from pathlib import Path

MANIFEST = Path("aws_cloud/manifest.json")

def load_manifest():

    if MANIFEST.exists():
        with open(MANIFEST) as f:
            return json.load(f)
    return {}

def save_manifest(data):

    with open(MANIFEST, "w") as f:
        json.dump(data, f, indent=4)

