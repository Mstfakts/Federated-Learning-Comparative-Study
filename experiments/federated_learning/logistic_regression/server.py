from flwr.server import ServerConfig
from flwr.server import start_server

from configs.config import config
from experiments.federated_learning.strategy import FedCustom
from utils.data_compatibility_utils import unflatten_dict
from utils.federated_learning_utils import weighted_average
from utils.reporting_utils import print_classification_report_from_dict

# Define the federated learning strategy
strategy = FedCustom(
    min_available_clients=2,
    fit_metrics_aggregation_fn=weighted_average,
    evaluate_metrics_aggregation_fn=weighted_average,
)

# Define the server configuration
server_config = ServerConfig(num_rounds=config['round'])

# Start the Flower server
hist = start_server(
    server_address=config['server']['address'],
    config=server_config,
    strategy=strategy,
)

test_sonuclari_epoch = {}
for k, v in hist.metrics_distributed.items():
    test_sonuclari_epoch[k] = v[-1][1]

unflattened_results = unflatten_dict(test_sonuclari_epoch)
print_classification_report_from_dict(unflattened_results)
