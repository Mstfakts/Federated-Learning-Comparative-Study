import os

from flwr.server import ServerConfig
from flwr.server import start_server

from configs.config import get_config
from src.federated.clients.xgboosts import average_dict
from utils.federated_learning_utils import create_strategy
from utils.reporting import print_classification_report_from_dict, unflatten_dict

config = get_config()

# Define the federated learning strategy
strategy = create_strategy(
    model_name=os.environ["config_file"],
    strategy_name=config['aggregation']['method']
)

# Define the server configuration
server_config = ServerConfig(num_rounds=config['round'])

# Start the Flower server
hist = start_server(
    server_address=config['server']['address'],
    config=server_config,
    strategy=strategy,
)

if os.environ["config_file"] != "xgboosts":
    test_result_for_each_epoch = {}
    for k, v in hist.metrics_distributed.items():
        test_result_for_each_epoch[k] = v[-1][1]

    unflattened_results = unflatten_dict(test_result_for_each_epoch)
else:
    unflattened_results = average_dict(hist.metrics_distributed['record'])

print_classification_report_from_dict(unflattened_results)
