from functools import reduce
from logging import WARNING
from typing import Dict, Optional, Union
from typing import List, Tuple
import logging
import math
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


class FairFed(FedAvg):

    def __repr__(self) -> str:
        return "FairFed"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inplace = False
        self.fairness_budget = 1.0

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

        if self.inplace:  # TODO inplace ksımı gereksizse kaldırılsın
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            aggregated_ndarrays = self.custom_aggregate(results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def custom_aggregate(self, results: List[Tuple[NDArrays, int]]) -> NDArrays:

        # 1) Toplam örnek sayısı ve client verilerinin toplanması
        weights_results = []
        total_examples = 0
        client_edos = {}
        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            n = fit_res.num_examples

            i=0
            while i < 5:
                key = f"equal.opportunity.difference_{i}"
                if key in fit_res.metrics:
                    eod = fit_res.metrics[key]
                    if i not in client_edos.keys():
                        client_edos[i] = eod
                    break
                i += 1
            weights_results.append((ndarrays, n, eod))
            total_examples += n

        # 2) Global EOD (ağırlıklı ortalama)
        eods = [eod for _, n, eod in weights_results]
        nums = [n for _, n, _ in weights_results]
        eod_global = sum(e * n for e, n in zip(eods, nums)) / total_examples

        # 3) FedAvg temel ağırlıkları
        base_weights = [n / total_examples for n in nums]
        print("G_eod: ", eod_global)

        # 4) Exponential reweighting
        raw_weights = [
            b * math.exp(-self.fairness_budget * abs(eod_i - eod_global))
            for b, eod_i in zip(base_weights, eods)
        ]
        Z = sum(raw_weights)
        norm_weights = [w / Z for w in raw_weights]

        client_edos = dict(sorted(client_edos.items(), key=lambda x: x[0]))
        print(client_edos)
        print(raw_weights)

        # 5) Katman katman ağırlıklı ortalama
        num_layers = len(weights_results[0][0])
        aggregated: NDArrays = []
        for layer_idx in range(num_layers):
            # her client’ın aynı katman indeksi için güncellemeyi alıp norm_weights ile çarp
            layer_updates = [
                client_ndarrays[layer_idx] * w
                for (client_ndarrays, _, _), w in zip(weights_results, norm_weights)
            ]
            aggregated.append(reduce(np.add, layer_updates))
        return aggregated
