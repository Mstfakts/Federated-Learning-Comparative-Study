import os

import yaml

current_file_directory = os.path.dirname(os.path.abspath(__file__))

with open(f"{current_file_directory}/mlp_config.yaml", "r") as file:
    config = yaml.safe_load(file)
