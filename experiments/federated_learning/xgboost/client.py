import argparse
import os
from logging import INFO
from typing import Dict

os.environ["config_file"] = "xgboost"
import flwr as fl
import xgboost as xgb
from flwr.client import start_client
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)
from flwr.common.logger import log
from sklearn.metrics import classification_report

from configs.config import config
from data.dataset import load_xgboost_data
from utils.data_compatibility_utils import flatten_dict


def parse_args() -> int:
    """Parse command-line arguments to get partition ID."""
    parser = argparse.ArgumentParser(description="Flower Client for Federated Learning")
    parser.add_argument(
        "--partition-id",
        choices=range(0, config['client']),
        default=0,
        type=int,
        help="Partition ID for the dataset split.",
    )
    return parser.parse_args().partition_id


class XGBoostClient(fl.client.Client):
    """Flower client implementing federated learning for XGBoost."""

    def __init__(
            self,
            train_dmatrix: xgb.DMatrix,
            valid_dmatrix: xgb.DMatrix,
            num_train: int,
            num_val: int,
            num_local_round: int,
            params: Dict,
    ) -> None:
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def _local_boost(self, bst_input):
        # Update trees based on local training data.
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for sever aggregation
        bst = bst_input[
              bst_input.num_boosted_rounds()
              - self.num_local_round: bst_input.num_boosted_rounds()
              ]

        return bst

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            # First round local training
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )
        else:
            bst = xgb.Booster(params=self.params)
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            bst.load_model(global_model)

            # Local training
            bst = self._local_boost(bst)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        bst = xgb.Booster(params=self.params)
        for para in ins.parameters.tensors:
            para_b = bytearray(para)
        bst.load_model(para_b)

        # Run evaluation
        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

        global_round = ins.config["global_round"]
        log(INFO, f"AUC = {auc} at round {global_round}")

        # Generate classification report
        y_pred_probs = bst.predict(self.valid_dmatrix)
        y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_probs]
        y_true = self.valid_dmatrix.get_label()

        report = classification_report(y_true, y_pred, output_dict=True)
        report = flatten_dict(report)
        report["AUC"] = auc

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_val,
            metrics=report,
        )


def main() -> None:
    partition_id = parse_args()

    train_data, test_data, valid_data, num_train, num_test, num_val = load_xgboost_data(
        partition_id=partition_id,
        n_partitions=config['client'],
        batch_size=config['data']['batch_size'],
        scale=config['data']['scale'],
        smote=config['data']['smote'],
    )

    params = {
        "objective": config['model']['objective'],
        "learning_rate": config['model']['learning_rate'],
        "max_depth": config['model']['max_depth'],
        "eval_metric": config['model']['eval_metric'],
        "num_parallel_tree": config['model']['num_parallel_tree'],
        "tree_method": config['model']['tree_method'],
    }

    client = XGBoostClient(
        train_data,
        valid_data,
        num_train,
        num_val,
        config['round'],
        params,
    )

    start_client(
        server_address=config['server']['address'],
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
