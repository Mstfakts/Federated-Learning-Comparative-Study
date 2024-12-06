from typing import List, Tuple, Dict

import numpy as np
from flwr.common import Metrics, Scalar
from sklearn.svm import LinearSVC

from src.federated.clients.xgboosts import evaluate_metrics_aggregation, config_func
from src.federated.custom_strategies.fedavg import FedAvg
from src.federated.custom_strategies.fedf1 import FedF1
from src.federated.custom_strategies.fedxgbbagging_fedavg_fedf1 import FedXgbBagging


def set_initial_params(model: LinearSVC, n_features: int, n_classes: int):
    """Set initial parameters as zeros.

    Required since model params are uninitialized until model.fit is called but server
    asks for initial parameters from clients at launch.
    """
    model.classes_ = np.array([i for i in range(n_classes)])
    model.coef_ = np.zeros((1, n_features))
    if hasattr(model, 'fit_intercept') and model.fit_intercept is not None:
        model.intercept_ = np.zeros(1)
    return model


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Scalar]:
    """Compute weighted average.

    It is generic implementation that averages only over floats and ints and drops the
    other data types of the Metrics.
    """
    print(metrics)
    # num_samples_list can represent number of sample or batches depending on the client
    num_samples_list = [n_batches for n_batches, _ in metrics]
    num_samples_sum = sum(num_samples_list)
    metrics_lists: Dict[str, List[float]] = {}
    for num_samples, all_metrics_dict in metrics:
        #  Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            if isinstance(value, (float, int)):
                metrics_lists[single_metric] = []
        # Just one iteration needed to initialize the keywords
        break

    for num_samples, all_metrics_dict in metrics:
        # Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            # Add weighted metric
            if isinstance(value, (float, int)):
                metrics_lists[single_metric].append(float(num_samples * value))

    weighted_metrics: Dict[str, Scalar] = {}
    for metric_name, metric_values in metrics_lists.items():
        weighted_metrics[metric_name] = sum(metric_values) / num_samples_sum

    return weighted_metrics


def create_strategy(model_name, strategy_name):
    if strategy_name == "FedF1":

        if model_name == "xgboosts":
            return FedXgbBagging(
                strategy="FedF1",
                fraction_fit=1.0,
                min_fit_clients=2,
                min_available_clients=2,
                min_evaluate_clients=2,
                fraction_evaluate=1.0,
                evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
                on_evaluate_config_fn=config_func,
                on_fit_config_fn=config_func,
            )

        return FedF1(
            min_available_clients=2,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

    elif strategy_name == "FedAvg":

        if model_name == "xgboosts":
            return FedXgbBagging(
                strategy="FedAvg",
                fraction_fit=1.0,
                min_fit_clients=2,
                min_available_clients=2,
                min_evaluate_clients=2,
                fraction_evaluate=1.0,
                evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
                on_evaluate_config_fn=config_func,
                on_fit_config_fn=config_func,
            )

        return FedAvg(
            min_available_clients=2,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

    else:
        raise ValueError(f"Unsupported strategy: {strategy_name}")
