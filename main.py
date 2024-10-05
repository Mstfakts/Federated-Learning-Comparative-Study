import os
import random

import numpy as np
import torch

from configs.config import config


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
    set_seed(random.randint(1, 1000))

    commands = [
        ["python", f"./{config['experiment']['folder']}/server.py"],
        ["python", f"./{config['experiment']['folder']}/client.py", "--partition-id", "0"],
        ["python", f"./{config['experiment']['folder']}/client.py", "--partition-id", "1"],
        ["python", f"./{config['experiment']['folder']}/client.py", "--partition-id", "2"],
        ["python", f"./{config['experiment']['folder']}/client.py", "--partition-id", "3"],
        ["python", f"./{config['experiment']['folder']}/client.py", "--partition-id", "4"]
    ]

    for command in commands:
        run_in_new_terminal(command)
