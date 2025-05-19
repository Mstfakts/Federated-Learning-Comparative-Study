import numpy as np
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

    @classmethod
    def set_initial_params(cls, model, n_features: int, n_classes: int):
        """Set initial parameters as zeros.

        Required since model params are uninitialized until model.fit is called but server
        asks for initial parameters from clients at launch.
        """

        if isinstance(model, MLPClassifier):
            X_dummy = np.zeros((2, n_features))
            y_dummy = np.array([0, 1])
            model.partial_fit(X_dummy, y_dummy, classes=np.array([0, 1]))
        else:
            model.classes_ = np.array([i for i in range(n_classes)])
            model.coef_ = np.zeros((1, n_features))
            if hasattr(model, 'fit_intercept') and model.fit_intercept is not None:
                model.intercept_ = np.zeros(1)
        return model
