from typing import Dict

from flwr.server import ServerConfig
from flwr.server.strategy import FedXgbBagging

from configs.config import config
from utils.reporting_utils import average_dict, print_classification_report_from_dict


def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = (
            sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    )
    record_aggredated = average_dict(eval_metrics)

    metrics_aggregated = {"AUC": auc_aggregated, "record": record_aggredated}
    return metrics_aggregated


def config_func(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    strategy = FedXgbBagging(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        min_evaluate_clients=2,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=config_func,
        on_fit_config_fn=config_func,
    )

    # Define the server configuration
    server_config = ServerConfig(num_rounds=config['round'])

    hist = start_server(
        server_address=config['server']['address'],
        config=server_config,
        strategy=strategy,
    )

    print_classification_report_from_dict(average_dict(hist.metrics_distributed['record']))
