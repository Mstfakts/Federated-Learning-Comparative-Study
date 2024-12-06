from sklearn.neural_network import MLPClassifier

from configs.config import get_config

config = get_config()

model = MLPClassifier(
    max_iter=config['model']['max_iter'],
    random_state=42
)
