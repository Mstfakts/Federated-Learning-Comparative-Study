from sklearn.ensemble import RandomForestClassifier

from configs.config import config

model = RandomForestClassifier(
    class_weight=config['model']['class_weight'],
    criterion=config['model']['criterion'],
    n_estimators=config['model']['n_estimators'],
    max_depth=config['model']['max_depth'],
    min_samples_split=config['model']['min_samples_split'],
    min_samples_leaf=config['model']['min_samples_leaf'],
    max_features=config['model']['max_features'],
    bootstrap=config['model']['bootstrap']
)
