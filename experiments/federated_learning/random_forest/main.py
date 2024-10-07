import os
import time

os.environ["config_file"] = "random_forest"
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
    for _ in range(10):
        set_seed(random.randint(1, 1000))

        commands = [
            ["python", f"./server.py"],
            ["python", f"./client.py", "--partition-id", "0"],
            ["python", f"./client.py", "--partition-id", "1"],
            ["python", f"./client.py", "--partition-id", "2"],
            ["python", f"./client.py", "--partition-id", "3"],
            ["python", f"./client.py", "--partition-id", "4"]
        ]

        for command in commands:
            run_in_new_terminal(command)
        time.sleep(40)
