import logging
import os

import yaml
from flwr.common.logger import log

# Root directory for accessing the configuration files
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    """
    Singleton class to manage configuration loading and caching.
    Loads configuration from a YAML file and caches the instance to avoid reloading.
    """
    _instances = {}

    def __new__(cls, config_name, *args, **kwargs):
        if config_name not in cls._instances:
            cls._instances[config_name] = super(Config, cls).__new__(cls, *args, **kwargs)
            config_path = ROOT_DIR + f'/configs/{config_name}_config.yaml'
            with open(config_path, 'r') as file:
                cls._instances[config_name].config = yaml.safe_load(file)
                log(
                    logging.INFO,
                    f"Current config file: {config_path}"
                )
        return cls._instances[config_name]


def get_config(config_name=None):
    """
    Load configuration based on the provided configuration name or environment variable.

    Args:
        config_name (str, optional): The name of the configuration file (without '_config.yaml').
                                    If None, the environment variable 'CONFIG_FILE' is used.

    Returns:
        dict: Loaded configuration data.
    """
    config_name = config_name if config_name else os.environ["config_file"]
    return Config(config_name).config
