import argparse
import os
import warnings
from typing import List, Tuple, Dict, Any, Optional

os.environ["config_file"] = "logistic_regression"
import numpy as np
from flwr.client import NumPyClient
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss

from configs.config import config
from data.dataset import load_data
from experiments.federated_learning.logistic_regression.model import model
from utils.data_compatibility_utils import flatten_dict
from utils.federated_learning_utils import set_initial_params

# Get partition ID from command-line arguments
parser = argparse.ArgumentParser(description="Flower Client")
parser.add_argument(
    "--partition-id",
    choices=range(0, config['client']),
    default=0,
    type=int,
    help="Partition ID (integer)",
)
args = parser.parse_args()
partition_id = args.partition_id

# Load data for the specified partition
train_loader, test_loader, val_loader, num_examples = load_data(
    partition_id=partition_id,
    n_partitions=config['client'],
    batch_size=config['data']['batch_size'],
    scale=config['data']['scale'],
    smote=config['data']['smote'],
)


# Define the Flower client
class FlowerClient(NumPyClient):

    def __init__(
            self,
            model,
            train_loader,
            test_loader,
            val_loader,
    ) -> None:
        self.model = model
        self.train_data = train_loader.dataset.features
        self.train_label = train_loader.dataset.labels
        self.val_data = val_loader.dataset.features
        self.val_label = val_loader.dataset.labels
        self.test_data = test_loader.dataset.features
        self.test_label = test_loader.dataset.labels

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

        # Set model parameters
        self.set_parameters(parameters)

        # Suppress convergence warnings for cleaner output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.train_data, self.train_label)

        # Compute training accuracy
        train_accuracy = self.model.score(self.train_data, self.train_label)

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


if __name__ == "__main__":
    from flwr.client import start_client

    # Initialize model parameters
    model = set_initial_params(model, n_features=23, n_classes=2)

    # Start Flower client
    client = FlowerClient(model, train_loader, test_loader, val_loader).to_client()
    start_client(
        server_address=config['server']['address'],
        client=client,
    )
