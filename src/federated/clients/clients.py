import abc
import logging
import warnings
from typing import List, Optional, Tuple, Dict, Any

import flwr as fl
import numpy as np
import xgboost as xgb
from fairlearn.metrics import equal_opportunity_difference
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
from sklearn.metrics import classification_report, log_loss

from utils.reporting import flatten_report


class BaseClient(fl.client.NumPyClient, abc.ABC):
    """Abstract server class enforcing algorithm-specific client behavior."""

    def __init__(self, model, train_loader, test_loader, val_loader, experiment_type, sensitive_features, client_id):
        super().__init__()
        self.model = model
        self.client_id = client_id
        self.experiment_type = experiment_type
        (self.X_train, self.y_train), (self.X_test, self.y_test), (self.X_val, self.y_val) = (
            (train_loader.dataset.features, train_loader.dataset.labels),
            (test_loader.dataset.features, test_loader.dataset.labels),
            (val_loader.dataset.features, val_loader.dataset.labels)
        )

        if self.experiment_type == "fairness_experiments":
            self.sensitive_test_features = test_loader.dataset.data[sensitive_features]
            self.sensitive_val_features = val_loader.dataset.data[sensitive_features]

    @abc.abstractmethod
    def get_parameters(self, config: Optional[fl.common.Config]) -> List[np.ndarray]:
        """Extract model weights for Flower."""
        ...

    @abc.abstractmethod
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model weights from Flower parameters."""
        ...

    @abc.abstractmethod
    def fit(self,
            parameters: List[np.ndarray],
            config: Optional[fl.common.Config] = None) -> Tuple[List[np.ndarray], int]:
        """Train on local data and return updated weights and stats."""
        ...

    @abc.abstractmethod
    def evaluate(self,
                 parameters: List[np.ndarray],
                 config: Optional[fl.common.Config] = None) -> Tuple[float, int, Dict[str, float]]:
        """Evaluate on local test data and return loss and metrics."""
        ...


class XGBoostClient(fl.client.Client):

    def __init__(self, model, train_loader, test_loader, val_loader, experiment_type, sensitive_features, client_id,
                 num_local_round, params):
        fl.client.Client.__init__(self)
        self.model = model
        self.train_dmatrix: xgb.DMatrix = train_loader
        self.test_dmatrix: xgb.DMatrix = test_loader
        self.valid_dmatrix: xgb.DMatrix = val_loader
        self.experiment_type = experiment_type
        self.client_id = client_id
        self.num_train = train_loader.num_row()
        self.num_test = test_loader.num_row()
        self.num_val = val_loader.num_row()
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
        report = flatten_report(val_report)
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
        log(logging.INFO, f"AUC = {auc} at round {global_round}")

        # Generate classification report
        y_pred_probs = bst.predict(self.valid_dmatrix)
        y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_probs]
        y_true = self.valid_dmatrix.get_label()

        report = classification_report(y_true, y_pred, output_dict=True)
        report = flatten_report(report)
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


class RandomForestClient(BaseClient):
    def get_parameters(self, config: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        params = [
            self.model.n_estimators,
            self.model.max_depth,
            self.model.min_samples_split,
            self.model.min_samples_leaf,
        ]
        return params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        if parameters:
            self.model.n_estimators = int(parameters[0])
            self.model.max_depth = int(parameters[1])
            self.model.min_samples_split = int(parameters[2])
            self.model.min_samples_leaf = int(parameters[3])

    def fit(
            self,
            parameters: List[np.ndarray],
            config: Optional[fl.common.Config] = None,
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        # Set model parameters
        self.set_parameters(parameters)

        # Suppress convergence warnings for cleaner output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        # Compute training accuracy
        train_accuracy = self.model.score(self.X_train, self.y_train)  # TODO accuracy_score() dene

        # Validate the model
        y_val_pred = self.model.predict(self.X_val)
        val_report = classification_report(
            self.y_val, y_val_pred, output_dict=True, zero_division=0
        )
        report = flatten_report(val_report)
        report["train_accuracy"] = train_accuracy

        return self.get_parameters(config), len(self.X_train), report

    def evaluate(
            self,
            parameters: List[np.ndarray],
            config: Optional[fl.common.Config] = None,
    ) -> Tuple[float, int, Dict[str, float]]:
        # Set model parameters
        self.set_parameters(parameters)

        # Predict probabilities for log loss computation
        y_pred_proba = self.model.predict(self.X_test)
        loss = log_loss(self.y_test, y_pred_proba)

        # Compute test accuracy
        test_accuracy = self.model.score(self.X_test, self.y_test)

        # Generate classification report
        y_test_pred = self.model.predict(self.X_test)
        test_report = classification_report(
            self.y_test, y_test_pred, output_dict=True, zero_division=0
        )
        report = flatten_report(test_report)
        report["test_accuracy"] = test_accuracy

        return loss, len(self.X_test), report


class MLPClient(BaseClient):
    def __init__(self, model, train_loader, test_loader, val_loader, experiment_type, sensitive_features, client_id):

        def get_coef_and_intercept_shapes(n_feature):
            """
            input_size: Özellik sayısı (örneğin, X.shape[1])
            hidden_layer_sizes: Gizli katman nöron sayılarını içeren liste/tupel (config['model']['hidden_layer_sizes'])
            output_size: Çıkış sayısı (örneğin, len(np.unique(y)))
            """
            input_size = n_feature
            hidden_layer_sizes = model.hidden_layer_sizes
            output_size = 1

            # Ağırlık (coefs_) şekillerini oluşturma
            coef_shapes = []
            # İlk ağırlık: giriş -> ilk gizli katman
            coef_shapes.append((input_size, hidden_layer_sizes[0]))
            # Aradaki gizli katmanlar: ardışık iki gizli katman arasındaki bağlantılar
            for i in range(len(hidden_layer_sizes) - 1):
                coef_shapes.append((hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))
            # Son ağırlık: son gizli katman -> çıkış katmanı
            coef_shapes.append((hidden_layer_sizes[-1], output_size))

            # Bias (intercept_) şekillerini oluşturma:
            intercept_shapes = []
            # Her gizli katman için bias vektörü
            for size in hidden_layer_sizes:
                intercept_shapes.append((size,))
            # Çıkış katmanı için bias vektörü
            intercept_shapes.append((output_size,))

            return [coef_shapes, intercept_shapes]

        super().__init__(model, train_loader, test_loader, val_loader, experiment_type, sensitive_features, client_id)
        self.shapes = get_coef_and_intercept_shapes(n_feature=self.X_train.shape[1])

    def get_parameters(self, config: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:

        # Check if the model has been initialized and has intercepts
        if hasattr(self.model, 'intercepts_') and self.model.intercepts_ is not None:
            params = [
                self.model.coefs_,
                self.model.intercepts_,
            ]
            flattened_list = [param.flatten() for sublist in params for param in sublist]
            params = np.concatenate(flattened_list).tolist()

        else:
            params = [
                self.model.coefs_,
            ]
        return params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # 2. Diziyi eski haline döndürmek için fonksiyon
        def reconstruct_arrays(combined_array, shapes):
            if np.all(combined_array == [0]) and len(self.model.intercepts_) == 1:
                return combined_array

            reconstructed = []
            index = 0
            for sublist_shapes in shapes:
                sublist = []
                for shape in sublist_shapes:
                    # Alt dizinin eleman sayısını hesapla
                    size = np.prod(shape)
                    # Alt diziyi yeniden şekillendir ve listeye ekle
                    sublist.append(combined_array[index:index + size].reshape(shape))
                    index += size
                reconstructed.append(sublist)
            return reconstructed

        # Convert parameters to a flat NumPy array
        flat_parameters = np.asarray(parameters)

        # Reconstruct the parameters from the flattened array
        reconstructed_parameters = reconstruct_arrays(flat_parameters, self.shapes)

        # Set the model's coefs_ and intercepts_
        self.model.coefs_ = reconstructed_parameters[0]
        if hasattr(self.model, 'intercepts_') and self.model.intercepts_ is not None:
            self.model.intercepts_ = reconstructed_parameters[1]

    def fit(
            self,
            parameters: List[np.ndarray],
            config: Optional[fl.common.Config] = None,
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:

        # Set model parameters
        self.set_parameters(parameters)

        # Suppress convergence warnings for cleaner output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        # Compute training accuracy
        train_accuracy = self.model.score(self.X_train, self.y_train)  # TODO accuracy_score() dene

        # Validate the model
        y_val_pred = self.model.predict(self.X_val)
        val_report = classification_report(
            self.y_val, y_val_pred, output_dict=True, zero_division=0
        )
        report = flatten_report(val_report)
        report["train_accuracy"] = train_accuracy

        return self.get_parameters(config), len(self.X_train), report

    def evaluate(
            self,
            parameters: List[np.ndarray],
            config: Optional[fl.common.Config] = None,
    ) -> Tuple[float, int, Dict[str, float]]:
        # Set model parameters
        self.set_parameters(parameters)

        # Predict probabilities for log loss computation
        y_pred_proba = self.model.predict(self.X_test)
        loss = log_loss(self.y_test, y_pred_proba)

        # Compute test accuracy
        test_accuracy = self.model.score(self.X_test, self.y_test)

        # Generate classification report
        y_test_pred = self.model.predict(self.X_test)
        test_report = classification_report(
            self.y_test, y_test_pred, output_dict=True, zero_division=0
        )
        report = flatten_report(test_report)
        report["test_accuracy"] = test_accuracy

        return loss, len(self.X_test), report


class LogisticRegressionClient(BaseClient):
    def get_parameters(self, config: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        if hasattr(self.model, 'fit_intercept') and self.model.fit_intercept is not None:
            params = [
                self.model.coef_,
                self.model.intercept_,
            ]
        else:
            params = [
                self.model.coef_,
            ]
        return params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Sunucudan aldığımız güncel parametreleri modele yükler.
        """
        self.model.coef_ = parameters[0]
        if hasattr(self.model, 'fit_intercept') and self.model.fit_intercept is not None:
            self.model.intercept_ = parameters[1]

    def fit(
            self,
            parameters: List[np.ndarray],
            config: Optional[fl.common.Config] = None,
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:

        # Set model parameters
        self.set_parameters(parameters)

        # Suppress convergence warnings for cleaner output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        # Compute training accuracy
        train_accuracy = self.model.score(self.X_train, self.y_train)  # TODO accuracy_score() dene

        # Validate the model
        y_val_pred = self.model.predict(self.X_val)
        val_report = classification_report(
            self.y_val, y_val_pred, output_dict=True, zero_division=0
        )
        report = flatten_report(val_report)
        report["train_accuracy"] = train_accuracy

        if self.experiment_type == "fairness_experiments":

            sensitive = self.sensitive_val_features.iloc[:, 0]
            sensitive = sensitive.to_numpy()

            eod = equal_opportunity_difference(
                y_true=self.y_val,
                y_pred=y_val_pred,
                sensitive_features=sensitive,
                method="between_groups"
            )

            log(logging.INFO, f"Client EODs: {eod}")

            unique_vals, counts = np.unique(sensitive, return_counts=True)
            log(logging.INFO, f"Gruplar ve adetleri: {dict(zip(unique_vals, counts))}")

            for a in unique_vals:
                mask = (sensitive == a)  # şimdi mask shape=(1062,), 1-boyutlu
                positives = np.sum(self.y_val[mask] == 1)
                tps = np.sum((self.y_val[mask] == 1) & (y_val_pred[mask] == 1))
                tpr = tps / positives if positives > 0 else float('nan')
                log(logging.INFO, f"Grup {a}: Pozitif={positives}, TP={tps}, TPR={tpr:.3f}")
            log(logging.INFO, f"Tahminlerde 0 sayısı: {np.sum(y_val_pred == 0)}")
            log(logging.INFO, f"Tahminlerde 1 sayısı: {np.sum(y_val_pred == 1)}")
            report[f"equal.opportunity.difference_{self.client_id}"] = eod

        return self.get_parameters(config), len(self.X_train), report

    def evaluate(
            self,
            parameters: List[np.ndarray],
            config: Optional[fl.common.Config] = None,
    ) -> Tuple[float, int, Dict[str, float]]:
        # Set model parameters
        self.set_parameters(parameters)

        # Predict probabilities for log loss computation
        y_pred_proba = self.model.predict(self.X_test)
        loss = log_loss(self.y_test, y_pred_proba)

        # Compute test accuracy
        test_accuracy = self.model.score(self.X_test, self.y_test)

        # Generate classification report
        y_test_pred = self.model.predict(self.X_test)
        test_report = classification_report(
            self.y_test, y_test_pred, output_dict=True, zero_division=0
        )
        report = flatten_report(test_report)
        report["test_accuracy"] = test_accuracy

        if self.experiment_type == "fairness_experiments":
            sensitive = self.sensitive_test_features.iloc[:, 0]
            sensitive = sensitive.to_numpy()

            y_test_pred = self.model.predict(self.X_test)
            eod = equal_opportunity_difference(
                y_true=self.y_test,
                y_pred=y_test_pred,
                sensitive_features=sensitive,
                method="between_groups"
            )

            report["EqualOpportunityDifference"] = eod
            report[f"equal.opportunity.difference_{self.client_id}"] = eod

        return loss, len(self.X_test), report


class LinearSVCClient(BaseClient):
    def get_parameters(self, config: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        if hasattr(self.model, 'fit_intercept') and self.model.fit_intercept is not None:
            params = [
                self.model.coef_,
                self.model.intercept_,
            ]
        else:
            params = [
                self.model.coef_,
            ]
        return params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Sunucudan aldığımız güncel parametreleri modele yükler.
        """
        self.model.coef_ = parameters[0]
        if hasattr(self.model, 'fit_intercept') and self.model.fit_intercept is not None:
            self.model.intercept_ = parameters[1]

    def fit(
            self,
            parameters: List[np.ndarray],
            config: Optional[fl.common.Config] = None,
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:

        # Set model parameters
        self.set_parameters(parameters)

        # Suppress convergence warnings for cleaner output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        # Compute training accuracy
        train_accuracy = self.model.score(self.X_train, self.y_train)  # TODO accuracy_score() dene

        # Validate the model
        y_val_pred = self.model.predict(self.X_val)
        val_report = classification_report(
            self.y_val, y_val_pred, output_dict=True, zero_division=0
        )
        report = flatten_report(val_report)
        report["train_accuracy"] = train_accuracy

        return self.get_parameters(config), len(self.X_train), report

    def evaluate(
            self,
            parameters: List[np.ndarray],
            config: Optional[fl.common.Config] = None,
    ) -> Tuple[float, int, Dict[str, float]]:
        # Set model parameters
        self.set_parameters(parameters)

        # Predict probabilities for log loss computation
        y_pred_proba = self.model.predict(self.X_test)
        loss = log_loss(self.y_test, y_pred_proba)

        # Compute test accuracy
        test_accuracy = self.model.score(self.X_test, self.y_test)

        # Generate classification report
        y_test_pred = self.model.predict(self.X_test)
        test_report = classification_report(
            self.y_test, y_test_pred, output_dict=True, zero_division=0
        )
        report = flatten_report(test_report)
        report["test_accuracy"] = test_accuracy

        return loss, len(self.X_test), report
