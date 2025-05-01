from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


class ModelFactory:
    _models = {
        "xgboost": XGBClassifier,
        "random_forest": RandomForestClassifier,
        "mlp": MLPClassifier,
        "logistic_regression": LogisticRegression,
        "linear_svc": LinearSVC,
    }

    @classmethod
    def available_models(cls):
        return list(cls._models.keys())

    @classmethod
    def create(cls, model_name: str, **kwargs):
        try:
            return cls._models[model_name](**kwargs)
        except KeyError:
            raise ValueError(f"Unknown model: {model_name}")