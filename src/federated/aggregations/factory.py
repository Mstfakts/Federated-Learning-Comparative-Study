from typing import List, Tuple, Dict

from flwr.common import Metrics, Scalar

from src.federated.aggregations.fairfed import FairFed
from src.federated.aggregations.fedavg import FedAvg
from src.federated.aggregations.fedf1 import FedF1
from src.federated.aggregations.fedxgbbagging_fedavg_fedf1 import FedXgbBagging
from utils.reporting import average_dict


class AggregationFactory:
    """
    Factory for creating federated aggregation strategies.
    """
    _strategies = {
        "FedAvg": FedAvg,
        "FedF1": FedF1,
        "FedXgbBagging": FedXgbBagging,
        "FairFed": FairFed
    }

    @classmethod
    def create(
            cls,
            model_name: str,
            strategy_name: str,
            **fdr_cfg,
    ):
        """
        Instantiate a federated aggregation strategy based on name.

        Args:
            model_name: The ML model name (e.g., 'xgboost').
            strategy_name: The aggregation name (e.g., 'FedAvg', 'FedF1', 'FedXgbBagging').
            fdr_cfg: Additional configuration (partitioner params, aggregation_fns, etc.).
        """
        # Handle XGBoost-specific strategy parameters
        params = fdr_cfg
        key = "FedXgbBagging" if model_name == "xgboost" else strategy_name
        try:
            Strategy = cls._strategies[key]
        except KeyError:
            raise ValueError(f"Unsupported aggregation strategy: {strategy_name}")

        return Strategy(**params)


def create_strategy(model_name: str, strategy_name: str):
    if model_name == "xgboost":
        aggregation_params = {
            "strategy": strategy_name,
            "fraction_fit": 1.0,
            "min_fit_clients": 2,
            "min_available_clients": 2,
            "min_evaluate_clients": 2,
            "fraction_evaluate": 1.0,
            "evaluate_metrics_aggregation_fn": evaluate_metrics_aggregation,
            "on_fit_config_fn": config_func,
            "on_evaluate_config_fn": config_func,
        }
    else:
        aggregation_params = {
            "min_available_clients": 2,
            "fit_metrics_aggregation_fn": weighted_average,
            "evaluate_metrics_aggregation_fn": weighted_average,
        }

    return AggregationFactory.create(
        model_name=model_name,
        strategy_name=strategy_name,
        **aggregation_params
    )


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


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Scalar]:
    """Compute weighted average.

    It is generic implementation that averages only over floats and ints and drops the
    other data types of the Metrics.
    """
    clients_results = {}
    for n, m in metrics:
        for k, v in m.items():
            if k not in clients_results:
                clients_results[k] = []
            else:
                clients_results[k].append(v)
    clients_results = dict(sorted(clients_results.items(), key=lambda x: (x[0], x[1])))
    for k, v in clients_results.items():
        print(k, v)

    # num_samples_list can represent number of sample or batches depending on the client
    num_samples_list = [n_batches for n_batches, _ in metrics]
    num_samples_sum = sum(num_samples_list)
    metrics_lists: Dict[str, List[float]] = {}
    for num_samples, all_metrics_dict in metrics:
        #  Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            if isinstance(value, (float, int)):
                if "equal.opportunity.difference" in single_metric:
                    for i in range(5):
                        single_metric = "equal.opportunity.difference_" + f"{str(i)}"
                        metrics_lists[single_metric] = []
                else:
                    metrics_lists[single_metric] = []
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
