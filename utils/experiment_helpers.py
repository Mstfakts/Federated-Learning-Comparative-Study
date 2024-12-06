import json
import logging
import os
import random
import time

import numpy as np
import torch
from flwr.common.logger import log

from configs.config import get_config
from utils.reporting import compute_averages, parse_experiment_data, parse_metrics


def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Parameters:
    seed (int): The seed value to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_algorithm_config(algorithm_name):
    """
    Load the configuration for the specified algorithm and set environment variables.

    Parameters:
    algorithm_name (str): The name of the machine learning algorithm to use.

    Returns:
    dict: The configuration dictionary for the specified algorithm.
    """
    config = get_config(algorithm_name)
    os.environ["config_file"] = algorithm_name
    return config


def start_commands(ml_algorithm):
    """
    Start the necessary commands for the given machine learning algorithm.

    Parameters:
    ml_algorithm (str): The name of the machine learning algorithm.
    """
    commands = [
        ["python", f"base/server.py"],
        ["python", f"clients/{ml_algorithm}.py", "--partition-id", "0"],
        ["python", f"clients/{ml_algorithm}.py", "--partition-id", "1"],
        ["python", f"clients/{ml_algorithm}.py", "--partition-id", "2"],
        ["python", f"clients/{ml_algorithm}.py", "--partition-id", "3"],
        ["python", f"clients/{ml_algorithm}.py", "--partition-id", "4"]
    ]

    for command in commands:
        os.system(f"start cmd /k {' '.join(command)}")


def wait_for_file(filepath, wait_interval=5):
    """
    Wait until a specific file is created.

    Parameters:
    file_path (str): The path of the file to wait for.
    wait_interval (int): The interval in seconds to wait before checking again (default is 5).
    """
    log(logging.INFO, "EXPERIMENT 1 is processing...")
    while not os.path.exists(filepath):
        time.sleep(wait_interval)
    log(logging.INFO, f"Created: {filepath}")


def wait_for_experiment_completion(filepath, target_count, iteration, wait_interval=5):
    """
    Check experiment results in a file until a target count is reached.

    Parameters:
    file_path (str): The path of the file containing experiment results.
    target_count (int): The target count for experiments and classes to reach.
    iteration (int): The current iteration number.
    wait_interval (int): The interval in seconds to wait before checking again (default is 5).
    """
    is_printed = False
    while True:
        experiment_count, class_count = count_experiments_and_classes(filepath)
        if experiment_count == target_count and class_count == target_count:
            log(logging.INFO, f"Completed. (Total EXPERIMENT: {experiment_count})")
            break
            # elif experiment_count == (iteration + 1) and class_count == (iteration + 1):
        elif experiment_count == (iteration + 1):
            time.sleep(wait_interval)
            break
        else:
            if not is_printed:
                log(logging.INFO, f"EXPERIMENT {(experiment_count + 1)} is processing...")
            is_printed = True
            time.sleep(wait_interval)


def count_experiments_and_classes(filepath):
    """
    Count the occurrences of "EXPERIMENT #1" and "Class 0" in the file.

    Parameters:
    file_path (str): The path of the file to read.

    Returns:
    tuple: A tuple containing the counts of experiments and classes found in the file.
    """
    experiment_count = 0
    class_count = 0

    with open(filepath, "r") as file:
        for line in file:
            if "EXPERIMENT #1:" in line:
                experiment_count += 1
            if "Class 0:" in line:
                class_count += 1

    return experiment_count, class_count


def compute_and_print_averages(filepath):
    """
    Compute and print the averages of experiment results.

    Parameters:
    file_path (str): The path of the file containing experiment results.
    """
    with open(filepath, 'r') as file:
        content = file.read()

    if os.environ["config_file"] == "xgboosts":
        averaged_metrics = parse_experiment_data(content)
        experiments_num = content.count("EXPERIMENT #")
    else:
        experiments = content.split("EXPERIMENT #")[1:]
        data = [parse_metrics(experiment) for experiment in experiments]
        experiments_num = len(data)
        averaged_metrics = compute_averages(*data)

    log_message = []
    log_message.append(" ")
    log_message.append("######################################")
    log_message.append(f"## Averages of {experiments_num} experiment trials: ##")
    log_message.append("######################################")
    log_message.append("--------------------------------------")

    for cls, metrics in averaged_metrics.items():
        log_message.append(f"{cls}:")
        for metric, avg in metrics.items():
            log_message.append(f"  {metric}: {avg}")
        log_message.append("--------------------------------------")

    log_message.append("\n")
    log_message.append("##############################")
    log_message.append(f"## Configuration Settings: ##")
    log_message.append("##############################")
    configs = get_config()
    configs["algorithm"] = os.environ["config_file"]
    config_str = json.dumps(configs, indent=4)
    log_message.append(config_str)

    final_log_message = "\n".join(log_message)
    log(logging.INFO, final_log_message)
