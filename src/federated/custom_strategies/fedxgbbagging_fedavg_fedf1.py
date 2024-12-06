import json
from logging import WARNING
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from flwr.common import EvaluateRes, FitRes, Parameters, Scalar
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedXgbBagging(FedAvg):
    """Configurable FedXgbBagging strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
            self,
            strategy: str,
            evaluate_function: Optional[
                Callable[
                    [int, Parameters, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            **kwargs: Any,
    ):

        assert strategy in ["FedF1", "FedAvg"], "Only FedAvg and FedF1 are supported for XGBoost"
        self.strategy = strategy

        self.evaluate_function = evaluate_function
        self.global_model: Optional[bytes] = None
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedXgbBagging(accept_failures={self.accept_failures}, strategy={self.strategy})"
        return rep

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using bagging."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate all the client trees
        global_model = self.global_model
        for _, fit_res in results:
            update = fit_res.parameters.tensors

            # Adjust strategy key name
            strategy_client_weight = fit_res.metrics["1.0_f1-score"] \
                if self.strategy == "FedF1" else fit_res.num_examples

            for bst in update:
                global_model = aggregate(global_model, bst, client_weights=strategy_client_weight)

        self.global_model = global_model

        return (
            Parameters(tensor_type="", tensors=[cast(bytes, global_model)]),
            {},
        )

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics using average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return 0, metrics_aggregated

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_function is None:
            # No evaluation function provided
            return None
        eval_res = self.evaluate_function(server_round, parameters, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics


def aggregate(
        bst_prev_org: Optional[bytes],
        bst_curr_org: bytes,
        client_weights: int = 1
) -> bytes:
    """Conduct bagging aggregation for given trees."""
    if not bst_prev_org:
        bst_curr = json.loads(bytearray(bst_curr_org))

        for idx, tree_curr in enumerate(bst_curr['learner']['gradient_booster']['model']['trees']):
            base_weights = [
                bw * client_weights
                for bw in tree_curr['base_weights']
            ]
            bst_curr['learner']['gradient_booster']['model']['trees'][idx] = base_weights

        return bst_curr_org

    # Get the tree numbers
    tree_num_prev, _ = _get_tree_nums(bst_prev_org)
    _, paral_tree_num_curr = _get_tree_nums(bst_curr_org)

    bst_prev = json.loads(bytearray(bst_prev_org))
    bst_curr = json.loads(bytearray(bst_curr_org))

    bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
        "num_trees"
    ] = str(tree_num_prev + paral_tree_num_curr)
    iteration_indptr = bst_prev["learner"]["gradient_booster"]["model"][
        "iteration_indptr"
    ]
    bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(
        iteration_indptr[-1] + paral_tree_num_curr
    )

    # Aggregate new trees
    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
    for tree_count in range(paral_tree_num_curr):
        base_weights = [
            bw * client_weights
            for bw in trees_curr[tree_count]['base_weights']
        ]
        trees_curr[tree_count]['base_weights'] = base_weights

        trees_curr[tree_count]["id"] = tree_num_prev + tree_count
        bst_prev["learner"]["gradient_booster"]["model"]["trees"].append(
            trees_curr[tree_count]
        )
        bst_prev["learner"]["gradient_booster"]["model"]["tree_info"].append(0)

    bst_prev_bytes = bytes(json.dumps(bst_prev), "utf-8")

    return bst_prev_bytes


def _get_tree_nums(xgb_model_org: bytes) -> Tuple[int, int]:
    xgb_model = json.loads(bytearray(xgb_model_org))
    # Get the number of trees
    tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_trees"
        ]
    )
    # Get the number of parallel trees
    paral_tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_parallel_tree"
        ]
    )
    return tree_num, paral_tree_num
