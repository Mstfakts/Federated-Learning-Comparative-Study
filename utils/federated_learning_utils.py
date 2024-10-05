from typing import List, Tuple, Dict

import numpy as np
from flwr.common import Metrics, Scalar
from sklearn.svm import LinearSVC


def set_initial_params(model: LinearSVC, n_classes: int, n_features: int, inter, coef):
    """Set initial parameters as zeros.

    Required since model params are uninitialized until model.fit is called but server
    asks for initial parameters from clients at launch.
    """
    model.classes_ = np.array([i for i in range(n_classes)])
    if n_classes > 2:
        # Çok sınıflı durumlarda, coefs_ boyutu (n_classes, n_features) olur
        model.coefs_ = np.zeros((n_classes, n_features))
        if model.intercepts_:
            model.intercepts_ = np.zeros(n_classes)
    else:
        # İkili sınıflandırmada, coefs_ boyutu (1, n_features) olur
        model.coefs_ = coef
        if model.intercepts_:
            model.intercepts_ = inter


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
