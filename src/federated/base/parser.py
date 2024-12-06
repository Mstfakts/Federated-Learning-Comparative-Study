import argparse

from configs.config import get_config

config = get_config()
# Get partition ID from command-line arguments
parser = argparse.ArgumentParser(description="Flower Client")
parser.add_argument(
    "--partition-id",
    choices=range(0, config['client']),
    default=0,
    type=int,
    help="Partition ID (integer)",
)
