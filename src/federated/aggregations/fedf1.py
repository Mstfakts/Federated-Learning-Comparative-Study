from functools import reduce
from logging import WARNING
from typing import Dict, Optional, Union
from typing import List, Tuple

import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
)
from flwr.common import NDArrays, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate_inplace

from configs.config import get_config

config = get_config()


class FedF1(FedAvg):

    def __repr__(self) -> str:
        return "FedF1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inplace = False

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            aggregated_ndarrays = custom_aggregate(results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated


def custom_aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    parametre1 = '1_f1-score'
    weights_results = [
        (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics)
        for i, (_, fit_res) in enumerate(results)
    ]

    # Calculate the total number of examples used during training
    num_param_total1 = sum(dict_[parametre1] for (_, dict_) in weights_results)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * metric[parametre1] for layer in weights] for weights, metric in weights_results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_param_total1
        for layer_updates in zip(*weighted_weights)
    ]

    return weights_prime
