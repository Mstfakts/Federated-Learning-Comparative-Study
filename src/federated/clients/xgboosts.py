import time
from logging import INFO
from typing import Dict

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
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from models.xgboost_params import params
from configs.config import get_config
from data.dataset import load_dmatrix
from src.federated.base.parser import parser
from utils.reporting import average_dict, flatten_dict


class XGBoostClient(fl.client.Client):
    """Flower client implementing federated learning for XGBoost."""

    def __init__(
            self,
            train_dmatrix: xgb.DMatrix,
            valid_dmatrix: xgb.DMatrix,
            test_dmatrix: xgb.DMatrix,
            num_train: int,
            num_val: int,
            num_test: int,
            num_local_round: int,
            params: Dict,
            sleep_sec: int = 0
    ) -> None:
        self.model = None
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.test_dmatrix = test_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.num_local_round = num_local_round
        self.params = params
        self.sleep_sec = sleep_sec

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def set_parameters(self, ins: FitIns):
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            bst = None
        else:
            bst = xgb.Booster(params=self.params)
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            bst.load_model(global_model)
        self.model = bst

    def _local_boost(self):

        if not self.model:
            # This case is when int(ins.config["global_round"]) is 1
            # First round local training
            bst_fit = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )
        else:
            # Update trees based on local training data.
            for i in range(self.num_local_round):
                self.model.update(self.train_dmatrix, self.model.num_boosted_rounds())

            # Bagging: extract the last N=num_local_round trees for sever aggregation
            bst_fit = self.model[
                      self.model.num_boosted_rounds()
                      - self.num_local_round: self.model.num_boosted_rounds()
                      ]

        self.model = bst_fit

    def fit(self, ins: FitIns) -> FitRes:
        time.sleep(self.sleep_sec)

        # Set model parameters
        self.set_parameters(ins)

        # Fit the model
        self._local_boost()

        # Compute training accuracy
        y_pred_probs = self.model.predict(self.train_dmatrix)
        y_train_pred = [1 if prob > 0.5 else 0 for prob in y_pred_probs]
        y_train_true = self.train_dmatrix.get_label()
        train_accuracy = accuracy_score(y_train_true, y_train_pred)

        # Validate the model
        y_pred_probs = self.model.predict(self.valid_dmatrix)
        y_val_pred = [1 if prob > 0.5 else 0 for prob in y_pred_probs]
        val_report = classification_report(
            self.valid_dmatrix.get_label(), y_val_pred, output_dict=True, zero_division=0
        )
        report = flatten_dict(val_report)
        report["train_accuracy"] = train_accuracy

        # Save model
        local_model = self.model.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics=report,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        bst = xgb.Booster(params=self.params)
        para_b = None
        for para in ins.parameters.tensors:
            para_b = bytearray(para)
        bst.load_model(para_b)

        # Run evaluation
        eval_results = bst.eval_set(
            evals=[(self.test_dmatrix, "test")],
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
            num_examples=self.num_test,
            metrics=report,
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


def main() -> None:
    args = parser.parse_args()
    partition_id = args.partition_id
    config = get_config()

    train_data, test_data, valid_data, num_train, num_test, num_val = load_dmatrix(
        partition_id=partition_id,
        n_partitions=config['client'],
        batch_size=config['data']['batch_size'],
        scale=config['data']['scale'],
        use_smote=config['data']['smote'],
        use_rus=config['data']['rus'],
        encode=config['data']['encode'],
        n_pca_components=config['data']['pca']
    )

    client = XGBoostClient(
        train_dmatrix=train_data,
        valid_dmatrix=valid_data,
        test_dmatrix=test_data,
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        num_local_round=config['round'],
        params=params,
        sleep_sec=2
    )

    start_client(
        server_address=config['server']['address'],
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
