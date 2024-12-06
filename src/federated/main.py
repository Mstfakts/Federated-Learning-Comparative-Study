import datetime
import logging
import os
import random
import time

from utils.experiment_helpers import (
    set_seed,
    set_algorithm_config,
    start_commands,
    wait_for_file,
    wait_for_experiment_completion,
    compute_and_print_averages
)


class ML_ALGORITHMS:
    """
    Supported algorithms
    """
    LINEAR_SVC = "linear_svc"
    LOGISTIC_REGRESSION = "logistic_regression"
    MLP = "mlp"
    RANDOM_FOREST = "random_forest"
    XGBOOSTS = "xgboosts"


def create_file_names():
    CURR_TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Result file
    RESULT_FILENAME = f"results_{CURR_TIME}.txt"
    RESULT_FILEPATH = ROOT_DIR + f'/src/federated/' + RESULT_FILENAME
    os.environ["result_filepath"] = RESULT_FILEPATH

    # Logging file
    LOG_FILENAME = f"experiment_logs_{CURR_TIME}.log"
    os.environ["log_filename"] = LOG_FILENAME


def create_logger():
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Logger settings
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.environ["log_filename"])
        ]
    )


def main():
    _ = set_algorithm_config(os.environ["ml_algorithm"])
    EXPERIMENT_REPEAT_NUM = int(os.environ["experiment_repeat_num"])

    for i in range(EXPERIMENT_REPEAT_NUM):
        set_seed(random.randint(1, 1000))

        start_commands(os.environ["ml_algorithm"])
        if i == 0:
            wait_for_file(os.environ["result_filepath"])

        wait_for_experiment_completion(
            os.environ["result_filepath"], target_count=EXPERIMENT_REPEAT_NUM, iteration=i
        )

    compute_and_print_averages(os.environ["result_filepath"])


if __name__ == "__main__":

    algorithms_for_experiment = [
        ML_ALGORITHMS.LINEAR_SVC,
        ML_ALGORITHMS.LOGISTIC_REGRESSION,
        ML_ALGORITHMS.RANDOM_FOREST,
        ML_ALGORITHMS.XGBOOSTS,
        ML_ALGORITHMS.MLP
    ]
    for alg in algorithms_for_experiment:
        # Create result and logfile names
        create_file_names()

        # Adjust logging configurations
        create_logger()

        # How many times to repeat an experiment
        os.environ["experiment_repeat_num"] = "10"

        # With which algorithm the experiment will be performed
        os.environ["ml_algorithm"] = alg

        # Start experiment process
        main()

        # Wait for a while
        time.sleep(30)
