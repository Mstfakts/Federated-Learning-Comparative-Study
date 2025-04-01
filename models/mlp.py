from sklearn.neural_network import MLPClassifier

from configs.config import get_config

config = get_config()

model = MLPClassifier(hidden_layer_sizes=100,
                      activation='relu',
                      solver='adam',
                      alpha=0.0001,
                      learning_rate_init=0.001,
                      max_iter=config['model']['max_iter'],
                      random_state=42)
