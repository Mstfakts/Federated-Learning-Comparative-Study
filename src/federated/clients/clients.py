import abc
import warnings
from typing import List, Optional, Tuple, Dict, Any

import flwr as fl
import numpy as np
from sklearn.metrics import classification_report, log_loss

from utils.reporting import flatten_report


class BaseClient(fl.client.NumPyClient, abc.ABC):
    """Abstract server class enforcing algorithm-specific client behavior."""

    def __init__(self, model, train_loader, test_loader, val_loader, sleep_sec=2):
        super().__init__()
        self.model = model
        (self.X_train, self.y_train), (self.X_test, self.y_test), (self.X_val, self.y_val) = (
            (train_loader.dataset.features, train_loader.dataset.labels),
            (test_loader.dataset.features, test_loader.dataset.labels),
            (val_loader.dataset.features, val_loader.dataset.labels)
        )
        self.sleep_sec = sleep_sec

    @abc.abstractmethod
    def get_parameters(self, config: Optional[fl.common.Config]) -> List[np.ndarray]:
        """Extract model weights for Flower."""
        ...

    @abc.abstractmethod
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model weights from Flower parameters."""
        ...

    @abc.abstractmethod
    def fit(self,
            parameters: List[np.ndarray],
            config: Optional[fl.common.Config] = None) -> Tuple[List[np.ndarray], int]:
        """Train on local data and return updated weights and stats."""
        ...

    @abc.abstractmethod
    def evaluate(self,
                 parameters: List[np.ndarray],
                 config: Optional[fl.common.Config] = None) -> Tuple[float, int, Dict[str, float]]:
        """Evaluate on local test data and return loss and metrics."""
        ...


class XGBoostClient(BaseClient):
    def get_parameters(self, config):
        # XGBoost stores parameters in its booster
        booster = self.model.get_booster()
        return booster.save_raw()  # or custom serialization to list of arrays

    def set_parameters(self, parameters):
        # Load raw booster bytes
        self.model.load_model(bytearray(parameters))

    def fit(
            self,
            parameters: List[np.ndarray],
            config: Optional[fl.common.Config] = None,
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        self.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train)
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(
            self,
            parameters: List[np.ndarray],
            config: Optional[fl.common.Config] = None,
    ) -> Tuple[float, int, Dict[str, float]]:
        self.set_parameters(parameters)
        preds = self.model.predict(self.X_test)
        loss = float(((preds != self.y_test).sum()) / len(self.y_test))
        return loss, len(self.X_test), {"accuracy": 1 - loss}


class RandomForestClient(BaseClient):
    def get_parameters(self, config: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        params = [
            self.model.n_estimators,
            self.model.max_depth,
            self.model.min_samples_split,
            self.model.min_samples_leaf,
        ]
        return params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        if parameters:
            self.model.n_estimators = int(parameters[0])
            self.model.max_depth = int(parameters[1])
            self.model.min_samples_split = int(parameters[2])
            self.model.min_samples_leaf = int(parameters[3])

    def fit(
            self,
            parameters: List[np.ndarray],
            config: Optional[fl.common.Config] = None,
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        # Set model parameters
        self.set_parameters(parameters)

        # Suppress convergence warnings for cleaner output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        # Compute training accuracy
        train_accuracy = self.model.score(self.X_train, self.y_train)  # TODO accuracy_score() dene

        # Validate the model
        y_val_pred = self.model.predict(self.X_val)
        val_report = classification_report(
            self.y_val, y_val_pred, output_dict=True, zero_division=0
        )
        report = flatten_report(val_report)
        report["train_accuracy"] = train_accuracy

        return self.get_parameters(config), len(self.X_train), report

    def evaluate(
            self,
            parameters: List[np.ndarray],
            config: Optional[fl.common.Config] = None,
    ) -> Tuple[float, int, Dict[str, float]]:
        # Set model parameters
        self.set_parameters(parameters)

        # Predict probabilities for log loss computation
        y_pred_proba = self.model.predict(self.X_test)
        loss = log_loss(self.y_test, y_pred_proba)

        # Compute test accuracy
        test_accuracy = self.model.score(self.X_test, self.y_test)

        # Generate classification report
        y_test_pred = self.model.predict(self.X_test)
        test_report = classification_report(
            self.y_test, y_test_pred, output_dict=True, zero_division=0
        )
        report = flatten_report(test_report)
        report["test_accuracy"] = test_accuracy

        return loss, len(self.X_test), report


class MLPClient(BaseClient):
    def __init__(
            self,
            model_name: str,
            data: Tuple[
                Tuple[np.ndarray, np.ndarray],
                Tuple[np.ndarray, np.ndarray],
                Tuple[np.ndarray, np.ndarray],
            ],
            config: Dict[str, Any],
    ):

        def get_coef_and_intercept_shapes(n_feature):
            """
            input_size: Özellik sayısı (örneğin, X.shape[1])
            hidden_layer_sizes: Gizli katman nöron sayılarını içeren liste/tupel (config['model']['hidden_layer_sizes'])
            output_size: Çıkış sayısı (örneğin, len(np.unique(y)))
            """
            input_size = n_feature
            hidden_layer_sizes = self.model.config['mlp']['hidden_layer_sizes']
            output_size = 2

            # Ağırlık (coefs_) şekillerini oluşturma
            coef_shapes = []
            # İlk ağırlık: giriş -> ilk gizli katman
            coef_shapes.append((input_size, hidden_layer_sizes[0]))
            # Aradaki gizli katmanlar: ardışık iki gizli katman arasındaki bağlantılar
            for i in range(len(hidden_layer_sizes) - 1):
                coef_shapes.append((hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))
            # Son ağırlık: son gizli katman -> çıkış katmanı
            coef_shapes.append((hidden_layer_sizes[-1], output_size))

            # Bias (intercepts_) şekillerini oluşturma:
            intercept_shapes = []
            # Her gizli katman için bias vektörü
            for size in hidden_layer_sizes:
                intercept_shapes.append((size,))
            # Çıkış katmanı için bias vektörü
            intercept_shapes.append((output_size,))

            return [coef_shapes, intercept_shapes]

        (X_train, y_train), _, _ = data  # Train - Val - Test
        super().__init__(model_name, data, config)
        self.shapes = get_coef_and_intercept_shapes(n_feature=X_train.shape[1])

    def get_parameters(self, config: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:

        # Check if the model has been initialized and has intercepts
        if hasattr(self.model, 'intercepts_') and self.model.intercepts_ is not None:
            params = [
                self.model.coefs_,
                self.model.intercepts_,
            ]
            flattened_list = [param.flatten() for sublist in params for param in sublist]
            params = np.concatenate(flattened_list).tolist()

        else:
            params = [
                self.model.coefs_,
            ]
        return params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # 2. Diziyi eski haline döndürmek için fonksiyon
        def reconstruct_arrays(combined_array, shapes):
            if np.all(combined_array == [0]) and len(self.model.intercepts_) == 1:
                return combined_array

            reconstructed = []
            index = 0
            for sublist_shapes in shapes:
                sublist = []
                for shape in sublist_shapes:
                    # Alt dizinin eleman sayısını hesapla
                    size = np.prod(shape)
                    # Alt diziyi yeniden şekillendir ve listeye ekle
                    sublist.append(combined_array[index:index + size].reshape(shape))
                    index += size
                reconstructed.append(sublist)
            return reconstructed

        # Convert parameters to a flat NumPy array
        flat_parameters = np.asarray(parameters)

        # Reconstruct the parameters from the flattened array
        reconstructed_parameters = reconstruct_arrays(flat_parameters, self.shapes)

        # Set the model's coefs_ and intercepts_
        self.model.coefs_ = reconstructed_parameters[0]
        if hasattr(self.model, 'intercepts_') and self.model.intercepts_ is not None:
            self.model.intercepts_ = reconstructed_parameters[1]

    def fit(
            self,
            parameters: List[np.ndarray],
            config: Optional[fl.common.Config] = None,
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:

        # Set model parameters
        self.set_parameters(parameters)

        # Suppress convergence warnings for cleaner output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        # Compute training accuracy
        train_accuracy = self.model.score(self.X_train, self.y_train)  # TODO accuracy_score() dene

        # Validate the model
        y_val_pred = self.model.predict(self.X_val)
        val_report = classification_report(
            self.y_val, y_val_pred, output_dict=True, zero_division=0
        )
        report = flatten_report(val_report)
        report["train_accuracy"] = train_accuracy

        return self.get_parameters(config), len(self.X_train), report

    def evaluate(
            self,
            parameters: List[np.ndarray],
            config: Optional[fl.common.Config] = None,
    ) -> Tuple[float, int, Dict[str, float]]:
        # Set model parameters
        self.set_parameters(parameters)

        # Predict probabilities for log loss computation
        y_pred_proba = self.model.predict(self.X_test)
        loss = log_loss(self.y_test, y_pred_proba)

        # Compute test accuracy
        test_accuracy = self.model.score(self.X_test, self.y_test)

        # Generate classification report
        y_test_pred = self.model.predict(self.X_test)
        test_report = classification_report(
            self.y_test, y_test_pred, output_dict=True, zero_division=0
        )
        report = flatten_report(test_report)
        report["test_accuracy"] = test_accuracy

        return loss, len(self.X_test), report


class LogisticRegressionClient(BaseClient):
    def get_parameters(self, config: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        if hasattr(self.model, 'fit_intercept') and self.model.fit_intercept is not None:
            params = [
                self.model.coef_,
                self.model.intercept_,
            ]
        else:
            params = [
                self.model.coef_,
            ]
        return params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Sunucudan aldığımız güncel parametreleri modele yükler.
        """
        self.model.coef_ = parameters[0]
        if hasattr(self.model, 'fit_intercept') and self.model.fit_intercept is not None:
            self.model.intercept_ = parameters[1]

    def fit(
            self,
            parameters: List[np.ndarray],
            config: Optional[fl.common.Config] = None,
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:

        # Set model parameters
        self.set_parameters(parameters)

        # Suppress convergence warnings for cleaner output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        # Compute training accuracy
        train_accuracy = self.model.score(self.X_train, self.y_train)  # TODO accuracy_score() dene

        # Validate the model
        y_val_pred = self.model.predict(self.X_val)
        val_report = classification_report(
            self.y_val, y_val_pred, output_dict=True, zero_division=0
        )
        report = flatten_report(val_report)
        report["train_accuracy"] = train_accuracy

        return self.get_parameters(config), len(self.X_train), report

    def evaluate(
            self,
            parameters: List[np.ndarray],
            config: Optional[fl.common.Config] = None,
    ) -> Tuple[float, int, Dict[str, float]]:
        # Set model parameters
        self.set_parameters(parameters)

        # Predict probabilities for log loss computation
        y_pred_proba = self.model.predict(self.X_test)
        loss = log_loss(self.y_test, y_pred_proba)

        # Compute test accuracy
        test_accuracy = self.model.score(self.X_test, self.y_test)

        # Generate classification report
        y_test_pred = self.model.predict(self.X_test)
        test_report = classification_report(
            self.y_test, y_test_pred, output_dict=True, zero_division=0
        )
        report = flatten_report(test_report)
        report["test_accuracy"] = test_accuracy

        return loss, len(self.X_test), report


class LinearSVCClient(BaseClient):
    def get_parameters(self, config: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        if hasattr(self.model, 'fit_intercept') and self.model.fit_intercept is not None:
            params = [
                self.model.coef_,
                self.model.intercept_,
            ]
        else:
            params = [
                self.model.coef_,
            ]
        return params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Sunucudan aldığımız güncel parametreleri modele yükler.
        """
        self.model.coef_ = parameters[0]
        if hasattr(self.model, 'fit_intercept') and self.model.fit_intercept is not None:
            self.model.intercept_ = parameters[1]

    def fit(
            self,
            parameters: List[np.ndarray],
            config: Optional[fl.common.Config] = None,
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:

        # Set model parameters
        self.set_parameters(parameters)

        # Suppress convergence warnings for cleaner output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        # Compute training accuracy
        train_accuracy = self.model.score(self.X_train, self.y_train)  # TODO accuracy_score() dene

        # Validate the model
        y_val_pred = self.model.predict(self.X_val)
        val_report = classification_report(
            self.y_val, y_val_pred, output_dict=True, zero_division=0
        )
        report = flatten_report(val_report)
        report["train_accuracy"] = train_accuracy

        return self.get_parameters(config), len(self.X_train), report

    def evaluate(
            self,
            parameters: List[np.ndarray],
            config: Optional[fl.common.Config] = None,
    ) -> Tuple[float, int, Dict[str, float]]:
        # Set model parameters
        self.set_parameters(parameters)

        # Predict probabilities for log loss computation
        y_pred_proba = self.model.predict(self.X_test)
        loss = log_loss(self.y_test, y_pred_proba)

        # Compute test accuracy
        test_accuracy = self.model.score(self.X_test, self.y_test)

        # Generate classification report
        y_test_pred = self.model.predict(self.X_test)
        test_report = classification_report(
            self.y_test, y_test_pred, output_dict=True, zero_division=0
        )
        report = flatten_report(test_report)
        report["test_accuracy"] = test_accuracy

        return loss, len(self.X_test), report
