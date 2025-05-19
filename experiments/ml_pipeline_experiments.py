import os
import random
import time

from configs.config_loader import load_federated_config
from utils.experiment_helpers import (
    set_seed,
    start_commands,
    wait_for_file,
    wait_for_experiment_completion,
    compute_and_print_averages,
    ML_ALGORITHMS,
    create_file_names,
    create_logger
)

EXPERIMENT = "ml_pipeline_experiments"  # Do not change it

federated_config = load_federated_config()[EXPERIMENT]

REPEAT_NUM = federated_config["repeat_num"]  # How many times each experiment will be repeated (default 10)
DATASET_NAME = federated_config["dataset"]  # The dataset that will be used for experiment
CLIENT_NUM = federated_config["client"]  # Number of clients that will be used
ROUND_NUM = federated_config["round"]


def main(algorithm, result_filepath):
    for i in range(REPEAT_NUM):
        set_seed(random.randint(1, 1000))

        start_commands(algorithm, result_filepath, EXPERIMENT, DATASET_NAME, CLIENT_NUM, ROUND_NUM)
        if i == 0:
            wait_for_file(result_filepath)

        wait_for_experiment_completion(
            result_filepath, target_count=REPEAT_NUM, iteration=i
        )

    compute_and_print_averages(result_filepath, algorithm)


if __name__ == "__main__":

    algorithms_for_experiment = [
        ML_ALGORITHMS.LOGISTIC_REGRESSION,
        ML_ALGORITHMS.MLP,
        ML_ALGORITHMS.LINEAR_SVC,
        ML_ALGORITHMS.RANDOM_FOREST,
        ML_ALGORITHMS.XGBOOSTS,
    ]

    for index, alg in enumerate(algorithms_for_experiment):
        # Create result and logfile names
        RESULT_FILEPATH, LOG_FILEPATH = create_file_names(experiment_name=EXPERIMENT)

        # Adjust logging configurations
        create_logger(LOG_FILEPATH)

        # Start experiment process
        main(alg, RESULT_FILEPATH)

        # Wait for a while
        time.sleep(30)
        os.system("""osascript -e 'tell application "Terminal" to quit'""")
        time.sleep(5)
