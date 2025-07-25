from src.federated.clients.clients import (
    XGBoostClient, RandomForestClient, MLPClient, LogisticRegressionClient, LinearSVCClient
)


class ClientFactory:
    _client_map = {
        "xgboost": XGBoostClient,
        "random_forest": RandomForestClient,
        "mlp": MLPClient,
        "logistic_regression": LogisticRegressionClient,
        "linear_svc": LinearSVCClient,
    }

    @classmethod
    def available_clients(cls):
        return list(cls._client_map.keys())

    @classmethod
    def create(cls, model_name, **kwargs):
        try:
            client_cls = cls._client_map[model_name]
        except KeyError:
            raise ValueError(f"Unknown client type: {model_name}")

        return client_cls(**kwargs).to_client()
