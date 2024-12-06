import os

from configs.config import get_config

config = get_config()

if os.environ["config_file"] == "xgboosts":
    params = {
        "objective": config['model']['objective'],
        "learning_rate": config['model']['learning_rate'],
        "max_depth": config['model']['max_depth'],
        "tree_method": config['model']['tree_method'],
    }
else:
    params = {}
