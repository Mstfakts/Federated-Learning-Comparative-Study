from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss

from configs.config import get_config
from data.data_loader import partition_data_loader
from models.mlp import model
from src.federated.base.flower_client import FlowerClient
from src.federated.base.parser import parser
from utils.reporting import flatten_dict


# Define the Flower client
class MLPClient(FlowerClient):

    def __init__(
            self,
            model,
            train_loader,
            test_loader,
            val_loader,
            sleep_sec: int = 0
    ) -> None:
        super().__init__(model, train_loader, test_loader, val_loader, initialize=False, sleep_sec=sleep_sec)

        # Shapes for reconstructing the model parameters
        self.shapes = [
            [(config['model']['input_size'], config['model']['hidden_layer_sizes'][0]),
             (config['model']['hidden_layer_sizes'][0], config['model']['output_size'])],  # Shapes for coefs_
            [(config['model']['hidden_layer_sizes'][0],), (config['model']['output_size'],)]  # Shapes for intercepts_
        ]

    def evaluate(
            self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        # Set model parameters
        self.set_parameters(parameters)

        # Predict probabilities for log loss computation
        y_pred_proba = self.model.predict_proba(self.test_data)
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
            if np.all(combined_array == [0]) and len(model.intercepts_) == 1:
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


if __name__ == "__main__":
    from flwr.client import start_client

    args = parser.parse_args()
    partition_id = args.partition_id

    config = get_config()

    train_loader, test_loader, val_loader, num_examples = partition_data_loader(partition_id)

    # Initialize model parameters
    # We need to set coefs_ and intercepts_ before the first round
    X_sample = train_loader.dataset.features
    y_sample = train_loader.dataset.labels

    config['model']['input_size'] = train_loader.dataset.features.shape[1]

    # Initialize the model with a single sample to set up the parameters
    model.partial_fit(X_sample[:1], y_sample[:1], classes=np.unique(y_sample))

    # Start Flower client
    client = MLPClient(model, train_loader, test_loader, val_loader, 2).to_client()
    start_client(
        server_address=config['server']['address'],
        client=client,
    )
