import argparse
import logging

from flwr.server import ServerConfig
from flwr.server import start_server

from configs.config_loader import load_datasets_config, load_algorithms_config, load_federated_config
from utils.federated_learning_utils import create_strategy
from utils.reporting import average_dict
from utils.reporting import print_classification_report_from_dict, unflatten_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch Flower server with the specified algorithm and dataset"
    )
    parser.add_argument(
        "--algorithm", required=True,
        choices=load_algorithms_config().keys(),
        help="Which algorithm to run (must match keys in configs/algorithms.yaml)",
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=load_datasets_config().keys(),
        help="Which dataset to use (must match keys in configs/datasets.yaml)",
    )
    parser.add_argument(
        "--result-file", default=None,
        help="",
    )
    parser.add_argument(
        "--experiment-type", type=str, required=True,
        choices=["ml_pipeline_experiments", "class_holdout", "fairness"],
        help="Type of the experiment"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    fdr_cfg = load_federated_config()[args.experiment_type]

    # Define the federated learning strategy
    strategy = create_strategy(
        model_name=args.algorithm,
        strategy_name=fdr_cfg['aggregation']
    )

    # Define the server configuration
    server_config = ServerConfig(num_rounds=fdr_cfg['rounds'])

    # Start the Flower server
    hist = start_server(
        server_address=fdr_cfg['server'],
        config=server_config,
        strategy=strategy,
    )

    if args.algorithm != "xgboost":
        test_result_for_each_epoch = {}
        for k, v in hist.metrics_distributed.items():
            test_result_for_each_epoch[k] = v[-1][1]

        unflattened_results = unflatten_dict(test_result_for_each_epoch)
    else:
        unflattened_results = average_dict(hist.metrics_distributed['record'])

    print_classification_report_from_dict(unflattened_results, RESULT_FILEPATH=args.result_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
