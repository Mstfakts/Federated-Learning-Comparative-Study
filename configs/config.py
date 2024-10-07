import logging
import os

import yaml
from flwr.common.logger import log

root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fl_directory = root_directory + "/experiments/federated_learning"

config_files = {
    'mlp': '/mlp/mlp_config.yaml',
    'linear_svc': '/linear_svc/linear_svc_config.yaml',
    'logistic_regression': '/logistic_regression/logistic_regression_config.yaml',
    'xgboost': '/xgboost/xgboost_config.yaml',
    'random_forest': '/random_forest/random_forest_config.yaml'
}

config_file = config_files[os.environ["config_file"]]
file_path = f"{fl_directory}/{config_file}"

with open(file_path, "r") as file:
    config = yaml.safe_load(file)
    config['result_file'] = os.path.split(file_path)[0] + '/results.txt'

log(
    logging.WARNING,
    f"Current config file: {config_file}"
)
