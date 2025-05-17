import argparse

from flwr.client import start_client

from configs.config_loader import load_datasets_config, load_algorithms_config, load_federated_config
from data.dataloader import partition_data_loader
from models.factory import ModelFactory
from src.federated.clients.client_factory import ClientFactory


def parse_args():
    datasets = load_datasets_config().keys()
    algos = load_algorithms_config().keys()

    parser = argparse.ArgumentParser(description="Federated Learning Client Runner")
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=datasets,
        help="Which dataset config to use"
    )
    parser.add_argument(
        "--algorithm", type=str, required=True,
        choices=algos,
        help="Model name to use for training"
    )
    parser.add_argument(
        "--partition-id", type=int, required=True,
        help="Data partition ID"
    )
    parser.add_argument(
        "--clients", type=int, default=5,
        help="Number of federated clients"
    )
    parser.add_argument(
        "--experiment-type", type=str, required=True,
        choices=["ml_pipeline_experiments", "class_holdout", "fairness_experiments"],
        help="Type of the experiment"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ds_cfg = load_datasets_config()[args.dataset]
    algo_cfg = load_algorithms_config()[args.algorithm]
    fdr_cfg = load_federated_config()[args.experiment_type]

    # Load data partition
    train_loader, test_loader, val_loader, num_examples = partition_data_loader(
        args.partition_id,
        args.clients,
        ds_cfg,
        fdr_cfg
    )

    # Instantiate chosen model via Factory
    model = ModelFactory.create(
        args.algorithm,
        **algo_cfg
    )

    # Initialize federated parameters
    ModelFactory.set_initial_params(
        model,
        n_features=train_loader.dataset.features.shape[1],
        n_classes=2
    )

    # Create specialized Flower client and start
    client = ClientFactory.create(
        model_name=args.algorithm,
        model=model,
        client_id=args.partition_id,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        experiment_type=args.experiment_type,
        sensitive_features=ds_cfg["sensitive_features"]
    )

    start_client(
        server_address=fdr_cfg['server'],
        client=client
    )


if __name__ == "__main__":
    main()
