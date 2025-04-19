from sklearn.neural_network import MLPClassifier
import pprint
from configs.config import get_config
from logging import WARNING
from flwr.common.logger import log

config = get_config()

hidden_layer_sizes = tuple(x for x in config['model']['hidden_layer_sizes'])
model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                      activation=config['model']['activation'],
                      solver=config['model']['solver'],
                      alpha=0.0001,
                      learning_rate_init=config['model']['learning_rate_init'],
                      max_iter=config['model']['max_iter'],
                      warm_start=config['model']['warm_start'],
                      verbose=True,
                      early_stopping=False,
                      random_state=42)

params_str = pprint.pformat(model.get_params())

log(WARNING, "Model parameters:\n%s", params_str)
