import time
import warnings
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from flwr.client import NumPyClient
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss

from utils.reporting import flatten_dict


# Define the Flower client
class FlowerClient(NumPyClient):

    def __init__(
            self,
            model,
            train_data_loader,
            test_data_loader,
            val_data_loader,
            initialize: bool = False,
            sleep_sec: int = 0
    ) -> None:
        self.model = model

        self.train_data = train_data_loader.dataset.features
        self.train_label = train_data_loader.dataset.labels

        self.test_data = test_data_loader.dataset.features
        self.test_label = test_data_loader.dataset.labels

        self.val_data = val_data_loader.dataset.features
        self.val_label = val_data_loader.dataset.labels

        self.sleep_sec = sleep_sec

        if initialize:
            self.init_model()

    def fit(
            self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:

        """
        Train the model on local data.

        Args:
            parameters (List[np.ndarray]): Model parameters received from the server.
            config (dict): Configuration dictionary (unused).

        Returns:
            Tuple[List[np.ndarray], int, dict]: Updated model parameters, number of samples, and metrics.
        """
        time.sleep(self.sleep_sec)

        # Set model parameters
        self.set_parameters(parameters)

        # Suppress convergence warnings for cleaner output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.train_data, self.train_label)

        # Compute training accuracy
        train_accuracy = self.model.score(self.train_data, self.train_label)  # TODO accuracy_score() dene

        # Validate the model
        y_val_pred = self.model.predict(self.val_data)
        val_report = classification_report(
            self.val_label, y_val_pred, output_dict=True, zero_division=0
        )
        report = flatten_dict(val_report)
        report["train_accuracy"] = train_accuracy

        return self.get_parameters(config), len(self.train_data), report

    def evaluate(
            self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate the model on local test data.

        Args:
            parameters (List[np.ndarray]): Model parameters received from the server.
            config (dict): Configuration dictionary (unused).

        Returns:
            Tuple[float, int, dict]: Loss, number of samples, and metrics.
        """
        # Set model parameters
        self.set_parameters(parameters)

        # Predict probabilities for log loss computation
        y_pred_proba = self.model.predict(self.test_data)
        loss = log_loss(self.test_label, y_pred_proba)

        # Compute test accuracy
        test_accuracy = self.model.score(self.test_data, self.test_label)

        # Generate classification report
        y_test_pred = self.model.predict(self.test_data)
        test_report = classification_report(
            self.test_label, y_test_pred, output_dict=True, zero_division=0
        )
        report = flatten_dict(test_report)
        report["test_accuracy"] = test_accuracy

        return loss, len(self.test_data), report

    def get_parameters(self, config: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        """
        Return the model parameters as a list of NumPy arrays.

        Args:
            config (dict, optional): Configuration dictionary (unused).

        Returns:
            List[np.ndarray]: Model parameters (weights and biases).
        """
        # Check if the model has been initialized and has intercepts
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
        Set the model parameters from a list of NumPy arrays.

        Args:
            parameters (List[np.ndarray]): Model parameters to set.
        """
        # Set the model's coef_ and intercept_
        self.model.coef_ = parameters[0]
        if hasattr(self.model, 'fit_intercept') and self.model.fit_intercept is not None:
            self.model.intercept_ = parameters[1]

    def init_model(self):
        """
        Set initial parameters as zeros.

        Required since model params are uninitialized until model.fit is called but server
        asks for initial parameters from clients at launch.
        """
        n_features = self.train_data.shape[1]
        n_classes = self.train_label.unique()

        self.model.classes_ = np.array([i for i in range(n_classes)])
        self.model.coef_ = np.zeros((1, n_features))
        if hasattr(self.model, 'fit_intercept') and self.model.fit_intercept is not None:
            self.model.intercept_ = np.zeros(1)
