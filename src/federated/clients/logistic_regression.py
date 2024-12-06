from configs.config import get_config
from data.data_loader import partition_data_loader
from models.logistic_regression import model
from src.federated.base.flower_client import FlowerClient
from src.federated.base.parser import parser
from utils.federated_learning_utils import set_initial_params

if __name__ == "__main__":
    from flwr.client import start_client

    args = parser.parse_args()
    partition_id = args.partition_id

    config = get_config()

    train_loader, test_loader, val_loader, num_examples = partition_data_loader(partition_id)

    # Initialize model parameters
    model = set_initial_params(model, n_features=train_loader.dataset.features.shape[1], n_classes=2)

    # Start Flower client
    client = FlowerClient(model, train_loader, test_loader, val_loader, sleep_sec=5).to_client()
    start_client(
        server_address=config['server']['address'],
        client=client,
    )
