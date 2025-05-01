import os

os.environ["config_file"] = "logistic_regression"

import argparse
from configs.config import get_config
from data.data_loader import partition_data_loader
from models.factory import ModelFactory
from src.federated.clients.client_factory import ClientFactory
from utils.federated_learning_utils import set_initial_params
from flwr.client import start_client
from utils.experiment_helpers import set_algorithm_config


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Client Runner")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=ModelFactory.available_models(),
        help="Model name to use for training"
    )
    parser.add_argument(
        "--partition-id", type=int, required=True,
        help="Data partition ID"
    )
    parser.add_argument(
        "--sleep-sec", type=int, default=2,
        help="Seconds to sleep between rounds"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_algorithm_config(algorithm_name=args.model)
    config = get_config()

    # Load data partition
    train_loader, test_loader, val_loader, num_examples = partition_data_loader(args.partition_id)

    # Instantiate chosen model via Factory
    model = ModelFactory.create(
        args.model,
        **config['model']
    )

    # Initialize federated parameters
    model = set_initial_params(
        model,
        n_features=train_loader.dataset.features.shape[1],
        n_classes=config.get('n_classes', 2)
    )

    # Create specialized Flower client and start
    client = ClientFactory.create(
        model_name=args.model,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        sleep_sec=args.sleep_sec
    )
    start_client(
        server_address=config['server']['address'],
        client=client
    )


if __name__ == "__main__":
    main()
