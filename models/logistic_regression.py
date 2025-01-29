from sklearn.linear_model import LogisticRegression

from configs.config import get_config

config = get_config()

model = LogisticRegression(
    C=config['model']['C'],
    max_iter=config['model']['max_iter'],
    penalty=config['model']['penalty'],
    class_weight=config['model']['class_weight'],
    solver=config['model']['solver'],
    n_jobs=config['model']['n_jobs'],
    warm_start=config['model']['warm_start']
)
