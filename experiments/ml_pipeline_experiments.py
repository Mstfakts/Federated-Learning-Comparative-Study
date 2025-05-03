import datetime
import logging
import os
import random
import time

from utils.experiment_helpers import (
    set_seed,
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

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Result file
    RESULT_FILENAME = f"results_{CURR_TIME}.txt"
    RESULT_FILEPATH = ROOT_DIR + f'/results/ml_pipeline_experiments/' + RESULT_FILENAME

    # Logging file
    LOG_FILENAME = f"experiment_logs_{CURR_TIME}.txt"
    LOG_FILEPATH = ROOT_DIR + f'/results/ml_pipeline_experiments/' + LOG_FILENAME

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


def main(algorithm, experiment_repeat_num, RESULT_FILEPATH):

    for i in range(experiment_repeat_num):
        set_seed(random.randint(1, 1000))

        start_commands(algorithm, RESULT_FILEPATH)
        if i == 0:
            wait_for_file(RESULT_FILEPATH)

        wait_for_experiment_completion(
            RESULT_FILEPATH, target_count=experiment_repeat_num, iteration=i
        )

    compute_and_print_averages(RESULT_FILEPATH, algorithm)


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
        RESULT_FILEPATH, LOG_FILEPATH = create_file_names()

        # Adjust logging configurations
        create_logger(LOG_FILEPATH)

        # Start experiment process
        main(alg, 10, RESULT_FILEPATH)

        # Wait for a while
        time.sleep(30)
        os.system("""osascript -e 'tell application "Terminal" to quit'""")
        time.sleep(5)
