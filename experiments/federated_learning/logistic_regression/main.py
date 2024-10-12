import os
import time

os.environ["config_file"] = "logistic_regression"
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_in_new_terminal(command):
    os.system(f"start cmd /k {' '.join(command)}")


if __name__ == "__main__":

    for _ in range(2):
        set_seed(random.randint(1, 1000))

        os.system(f"start cmd /k {' '.join(['python', './server.py'])}")
        time.sleep(2)
        commands = [
            ["python", f"./client.py", "--partition-id", "0"],
            ["python", f"./client.py", "--partition-id", "1"],
            ["python", f"./client.py", "--partition-id", "2"],
            ["python", f"./client.py", "--partition-id", "3"],
            ["python", f"./client.py", "--partition-id", "4"]
        ]

        for command in commands:
            run_in_new_terminal(command)

        time.sleep(60)
