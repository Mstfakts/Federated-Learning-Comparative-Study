import os
import yaml

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CFG_DIR = os.path.join(_BASE, "configs")


def load_datasets_config() -> dict:
    path = os.path.join(_CFG_DIR, "datasets.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def load_algorithms_config() -> dict:
    path = os.path.join(_CFG_DIR, "models.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def load_federated_config() -> dict:
    path = os.path.join(_CFG_DIR, "federated.yaml")
    with open(path) as f:
        return yaml.safe_load(f)
