import logging
from functools import reduce
from typing import List, Tuple
from configs.config import config
import numpy as np
from flwr.common import NDArrays, parameters_to_ndarrays
from flwr.common.logger import log


def custom_aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    def weighted_average():
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for i, (_, fit_res) in enumerate(results)
        ]

        # Calculate the total number of examples used during training
        num_examples_total = sum(num_examples for (_, num_examples) in weights_results)

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in weights_results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime

    def train_precision():
        # Toplam ağırlıkları ve ağırlıklı parametreleri tutacak değişkenler
        total_precision = 0
        for result in results:
            _, fit_res = result
            total_precision += fit_res.metrics['precision']

        scales = [
            fit_res.metrics['precision'] / total_precision
            for _, fit_res in results
        ]
        log(logging.WARNING, f"Scales: {scales}")

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), scales[i])
            for i, (_, fit_res) in enumerate(results)
        ]

        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in weights_results
        ]

        num_examples_total = sum(num_examples for (_, num_examples) in weights_results)
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime

    def adaptive_aggregation():
        total_entropy = 0

        # 1. Adım: İstemcilerin entropilerini hesapla
        for result in results:
            _, fit_res = result
            precision = fit_res.metrics['precision']
            entropy = -precision * np.log(precision + 1e-10)  # Yüksek precision -> düşük entropi
            total_entropy += entropy

        # 2. Adım: Ağırlıkları entropi ile ölçeklendir
        scales = [
            (1 - (-fit_res.metrics['precision'] * np.log(fit_res.metrics['precision'] + 1e-10)) / total_entropy)
            for _, fit_res in results
        ]

        # 3. Adım: İstemci parametrelerine bu ağırlıkları uygula
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), scales[i])
            for i, (_, fit_res) in enumerate(results)
        ]

        # 4. Adım: Tüm istemcilerin ağırlıklı parametrelerini topla
        weighted_weights = [
            [layer * scale for layer in weights]
            for weights, scale in weights_results
        ]

        aggregated_weights = [
            np.sum(layer_updates, axis=0) / len(results)
            for layer_updates in zip(*weighted_weights)
        ]

        return aggregated_weights

    def entropy_weighted_average():
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.entropy)
            for _, fit_res in results
        ]

        # Model güncellemelerini ve entropi değerlerini ayrı listelere ayır
        model_updates = [weights for (weights, _) in weights_results]
        entropies = [entropy for (_, entropy) in weights_results]

        # Bölme işleminde sıfıra bölmeyi önlemek için küçük bir epsilon değeri ekleyin
        epsilon = 1e-8

        # Entropilere dayalı ağırlıkları hesapla (entropinin tersi alınır)
        inverse_entropies = [1 / (entropy + epsilon) for entropy in entropies]

        # Ağırlıkları normalleştir (toplamları 1 olacak şekilde)
        total_inverse_entropy = sum(inverse_entropies)
        normalized_weights = [w / total_inverse_entropy for w in inverse_entropies]

        # Her bir model güncellemesini ilgili ağırlıkla çarp
        weighted_updates = [
            [layer * weight for layer in model_update]
            for model_update, weight in zip(model_updates, normalized_weights)
        ]

        # Tüm ağırlıklı güncellemeleri katman bazında topla
        weights_prime = [
            reduce(np.add, layer_updates)
            for layer_updates in zip(*weighted_updates)
        ]

        return weights_prime

    def metric():
        parametre0 = '0_f1-score'
        parametre1 = '1_f1-score'
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics)
            for i, (_, fit_res) in enumerate(results)
        ]

        # Calculate the total number of examples used during training
        # num_param_total0 = sum(dict_[parametre0] for (_, dict_) in weights_results)
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

    def entro_metric():
        parametre1 = '1_f1-score'
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics)
            for i, (_, fit_res) in enumerate(results)
        ]

        # Calculate the total number of examples used during training
        num_param_total1 = sum(dict_[parametre1] for (_, dict_) in weights_results)

        def calc_ent(m):
            epsilon = 1e-10
            entropy = -m * np.log2(m + epsilon) - (1 - m) * np.log2(1 - m + epsilon)
            weight = 1 / (entropy + epsilon)  # Entropik ağırlık
            return weight

        entropies = [calc_ent(metric[parametre1]) for _, metric in weights_results]
        total_weight = sum(entropies)
        normalized_weights = [w / total_weight for w in entropies]

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * normalized_weights[i] for layer in weights] for i, (weights, metric) in enumerate(weights_results)
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_param_total1
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime

    aggregation_method = {
        'weighted': weighted_average,
        'f1': metric
    }

    selected_method = aggregation_method[config['aggregation']['method']]
    log(
        logging.WARNING,
        f"The aggregation method is: {config['aggregation']['method']}"
    )

    weights_prime = selected_method()
    return weights_prime
