from typing import List, Dict, Any, Optional

import numpy as np

from data.data_loader import partition_data_loader
from configs.config import get_config
from src.federated.base.flower_client import FlowerClient
from src.federated.base.parser import parser
from models.random_forest import model


# Define the Flower client
class RFClient(FlowerClient):

    def get_parameters(self, config: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        params = [
            self.model.n_estimators,
            self.model.max_depth,
            self.model.min_samples_split,
            self.model.min_samples_leaf,
        ]
        return params

    def set_parameters(self, parameters: List[np.ndarray]):
        if parameters:
            self.model.n_estimators = int(parameters[0])
            self.model.max_depth = int(parameters[1])
            self.model.min_samples_split = int(parameters[2])
            self.model.min_samples_leaf = int(parameters[3])


if __name__ == "__main__":
    from flwr.client import start_client

    args = parser.parse_args()
    partition_id = args.partition_id

    config = get_config()

    train_loader, test_loader, val_loader, num_examples = partition_data_loader(partition_id)

    # It is for initial params
    X_sample = train_loader.dataset.features
    y_sample = train_loader.dataset.labels
    model.fit(X_sample[:10], y_sample[:10])

    # Start Flower client
    client = RFClient(model, train_loader, test_loader, val_loader, initialize=False, sleep_sec=1).to_client()
    start_client(
        server_address=config['server']['address'],
        client=client,
    )
