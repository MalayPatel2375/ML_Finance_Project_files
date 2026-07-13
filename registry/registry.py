import json
from datetime import datetime
from pathlib import Path

REGISTRY_PATH = Path("registry/model_registry.json")

def load_registry():

    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)

def save_registry(registry):

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)

def register_model(model_info):

    registry = load_registry()
    registry["models"].append(model_info)

    save_registry(registry)
    print(f"Model Registered Successfully.")

def get_latest_model():
    registry = load_registry()
    if not registry["models"]:
        return None
    return registry["models"][-1]