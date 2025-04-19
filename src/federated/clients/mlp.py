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


def get_coef_and_intercept_shapes():
    """
    input_size: Özellik sayısı (örneğin, X.shape[1])
    hidden_layer_sizes: Gizli katman nöron sayılarını içeren liste/tupel (config['model']['hidden_layer_sizes'])
    output_size: Çıkış sayısı (örneğin, len(np.unique(y)))
    """
    input_size = config['model']['input_size']
    hidden_layer_sizes = config['model']['hidden_layer_sizes']
    output_size = config['model']['output_size']

    # Ağırlık (coefs_) şekillerini oluşturma
    coef_shapes = []
    # İlk ağırlık: giriş -> ilk gizli katman
    coef_shapes.append((input_size, hidden_layer_sizes[0]))
    # Aradaki gizli katmanlar: ardışık iki gizli katman arasındaki bağlantılar
    for i in range(len(hidden_layer_sizes) - 1):
        coef_shapes.append((hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))
    # Son ağırlık: son gizli katman -> çıkış katmanı
    coef_shapes.append((hidden_layer_sizes[-1], output_size))

    # Bias (intercepts_) şekillerini oluşturma:
    intercept_shapes = []
    # Her gizli katman için bias vektörü
    for size in hidden_layer_sizes:
        intercept_shapes.append((size,))
    # Çıkış katmanı için bias vektörü
    intercept_shapes.append((output_size,))

    return [coef_shapes, intercept_shapes]


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
        self.shapes = get_coef_and_intercept_shapes()

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
    classes = np.unique(y_sample)
    init_indices = [np.where(y_sample == c)[0][0] for c in classes]
    X_init = X_sample[init_indices]
    y_init = y_sample[init_indices]
    model.partial_fit(X_init, y_init, classes=classes)

    # Start Flower client
    client = MLPClient(model, train_loader, test_loader, val_loader, 2).to_client()
    start_client(
        server_address=config['server']['address'],
        client=client,
    )
