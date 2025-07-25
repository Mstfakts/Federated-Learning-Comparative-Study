import datetime
import json
import logging
import os
import random
import time

import numpy as np
import torch
from flwr.common.logger import log

from configs.config_loader import load_algorithms_config
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


class ML_ALGORITHMS:
    """
    Supported algorithms
    """
    LINEAR_SVC = "linear_svc"
    LOGISTIC_REGRESSION = "logistic_regression"
    MLP = "mlp"
    RANDOM_FOREST = "random_forest"
    XGBOOSTS = "xgboost"


def create_file_names(experiment_name):
    CURR_TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Result file
    RESULT_FILENAME = f"results_{CURR_TIME}.txt"
    RESULT_FILEPATH = ROOT_DIR + f'/results/{experiment_name}/' + RESULT_FILENAME

    # Logging file
    LOG_FILENAME = f"experiment_logs_{CURR_TIME}.txt"
    LOG_FILEPATH = ROOT_DIR + f'/results/{experiment_name}/' + LOG_FILENAME

    return RESULT_FILEPATH, LOG_FILEPATH


def create_logger(LOG_FILEPATH):
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Logger settings
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILEPATH)
        ]
    )


def start_commands(ml_algorithm, result_filepath, experiment_type, dataset_name, client_num, round_num):
    """
    Start the necessary commands for the given machine learning algorithm.

    Parameters:
    ml_algorithm (str): The name of the machine learning algorithm.
    """

    # TODO hem Win hem MacOS için ayrı ayrı kod yaz
    conda_init_script = "/Users/mustafaaktas/anaconda3/etc/profile.d/conda.sh"
    env_name = "Federated-Learning-Comparative-Study"
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # TODO önce merkezi ile test edilmeli. Ardından FL sonucu alınıp çıktı dosyaya yazılmalı

    commands = [
        ["python",
         f"{ROOT_DIR}/src/federated/server/server.py",
         "--algorithm", f"{ml_algorithm}",
         "--dataset", f"{dataset_name}",
         "--result-file", f"{result_filepath}",
         "--experiment-type", f"{experiment_type}",
         "--rounds", f"{round_num}"],
        ["python",
         f"{ROOT_DIR}/src/federated/clients/client_main.py",
         "--algorithm", f"{ml_algorithm}",
         "--partition-id", "0",
         "--clients", f"{client_num}",
         "--dataset", f"{dataset_name}",
         "--experiment-type", f"{experiment_type}"],
        ["python",
         f"{ROOT_DIR}/src/federated/clients/client_main.py",
         "--algorithm", f"{ml_algorithm}",
         "--partition-id", "1",
         "--clients", f"{client_num}",
         "--dataset", f"{dataset_name}",
         "--experiment-type", f"{experiment_type}"],
        ["python",
         f"{ROOT_DIR}/src/federated/clients/client_main.py",
         "--algorithm", f"{ml_algorithm}",
         "--partition-id", "2",
         "--clients", f"{client_num}",
         "--dataset", f"{dataset_name}",
         "--experiment-type", f"{experiment_type}"],
        ["python",
         f"{ROOT_DIR}/src/federated/clients/client_main.py",
         "--algorithm", f"{ml_algorithm}",
         "--partition-id", "3",
         "--clients", f"{client_num}",
         "--dataset", f"{dataset_name}",
         "--experiment-type", f"{experiment_type}"],
        ["python",
         f"{ROOT_DIR}/src/federated/clients/client_main.py",
         "--algorithm", f"{ml_algorithm}",
         "--partition-id", "4",
         "--clients", f"{client_num}",
         "--dataset", f"{dataset_name}",
         "--experiment-type", f"{experiment_type}"]
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

    if algorithm == "xgboost":
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
