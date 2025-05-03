import json
import logging
import os
import random
import time

import numpy as np
import torch
from flwr.common.logger import log

from configs.config_loader import load_datasets_config, load_algorithms_config
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



def start_commands(ml_algorithm, RESULT_FILEPATH):
    """
    Start the necessary commands for the given machine learning algorithm.

    Parameters:
    ml_algorithm (str): The name of the machine learning algorithm.
    """

    # TODO hem Win hem MacOS için ayrı ayrı kod yaz
    conda_init_script = "/Users/mustafaaktas/anaconda3/etc/profile.d/conda.sh"
    env_name = "Federated-Learning-Comparative-Study"

    commands = [
        ["python", "/Users/mustafaaktas/PycharmProjects/Federated-Learning-Comparative-Study/src/federated/server/server.py", "--algorithm", f"{ml_algorithm}", "--dataset", "hmeq", "--rounds", "10", "--resultfile", f"{RESULT_FILEPATH}"],
        ["python", "/Users/mustafaaktas/PycharmProjects/Federated-Learning-Comparative-Study/src/federated/clients/client_main.py", "--algorithm", f"{ml_algorithm}", "--sleep-sec", "2", "--partition-id", "0", "--dataset", "hmeq", "--rounds", "10"],
        ["python", "/Users/mustafaaktas/PycharmProjects/Federated-Learning-Comparative-Study/src/federated/clients/client_main.py", "--algorithm", f"{ml_algorithm}", "--sleep-sec", "2", "--partition-id", "1", "--dataset", "hmeq", "--rounds", "10"],
        ["python", "/Users/mustafaaktas/PycharmProjects/Federated-Learning-Comparative-Study/src/federated/clients/client_main.py", "--algorithm", f"{ml_algorithm}", "--sleep-sec", "2", "--partition-id", "2", "--dataset", "hmeq", "--rounds", "10"],
        ["python", "/Users/mustafaaktas/PycharmProjects/Federated-Learning-Comparative-Study/src/federated/clients/client_main.py", "--algorithm", f"{ml_algorithm}", "--sleep-sec", "2", "--partition-id", "3", "--dataset", "hmeq", "--rounds", "10"],
        ["python", "/Users/mustafaaktas/PycharmProjects/Federated-Learning-Comparative-Study/src/federated/clients/client_main.py", "--algorithm", f"{ml_algorithm}", "--sleep-sec", "2", "--partition-id", "4", "--dataset", "hmeq", "--rounds", "10"]
    ]

    for command in commands:
        cmd_str = " ".join(command)

        final_command = (
            f'source {conda_init_script}; '
            f'conda activate {env_name}; '
            f'{cmd_str}'
        )
        os.system(f'''osascript -e 'tell application "Terminal" to do script "{final_command}"' ''')



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


def compute_and_print_averages(filepath, algorithm):
    """
    Compute and print the averages of experiment results.

    Parameters:
    file_path (str): The path of the file containing experiment results.
    """
    with open(filepath, 'r') as file:
        content = file.read()

    if algorithm == "xgboosts":
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

    configs = load_algorithms_config()[algorithm]
    config_str = json.dumps(configs, indent=4)
    log_message.append(config_str)

    final_log_message = "\n".join(log_message)
    log(logging.INFO, final_log_message)
