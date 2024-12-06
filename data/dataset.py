import logging
import os
import random
from typing import Dict, Tuple

import pandas as pd
import torch
import xgboost as xgb
from datasets import Dataset as HFDataset
from flwr.common.logger import log
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import Dataset, DataLoader

from configs.config import get_config
from data.data_process import (
    split_data,
    apply_smote,
    apply_rus,
    apply_pca,
    apply_scaling,
    apply_kbest,
    apply_encoding,
    data_cleaning,
    transform_dataset_to_dmatrix,
)

# Load configuration
config = get_config()

# Set up the data root directory
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))


# PyTorch Dataset sınıfı
class DataFrameDataset(Dataset):
    """
    Custom PyTorch Dataset class for loading features and labels from a DataFrame.

    Args:
        data_frame (pd.DataFrame): The DataFrame containing the data.
        use_pca (bool): Whether PCA was applied or not. If True, 'index' column may be omitted.
    """

    def __init__(self, data_frame: pd.DataFrame, use_pca: bool = False):
        self.data = data_frame.reset_index(drop=True)
        self.labels = self.data['def_pay'].values

        # Determine which columns to drop
        drop_columns = ['def_pay']
        if not use_pca:
            # If PCA is not used, ensure 'index' column is dropped if it exists
            if 'index' in self.data.columns:
                drop_columns.append('index')

        self.features = self.data.drop(columns=drop_columns).values

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label


def load_dataloader(
        partition_id: int,
        n_partitions: int,
        batch_size: int = 32,
        use_smote: bool = False,
        use_rus: bool = False,
        scale: bool = False,
        encode: bool = False,
        n_pca_components: int = 0,
        kbest: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Load and partition the dataset for federated learning (PyTorch).

    This function:
      - Loads data from a CSV file.
      - Cleans and optionally encodes the data.
      - Partitions the data using a Dirichlet partitioner.
      - Splits the data into train/validation/test sets.
      - Optionally applies scaling, SMOTE, RUS, and PCA transformations.
      - Creates PyTorch Datasets and DataLoaders from the processed data.

    Args:
        partition_id (int): The partition (client) ID.
        n_partitions (int): The total number of partitions (clients).
        batch_size (int): Batch size for DataLoader.
        use_smote (bool): Apply SMOTE to the training data if True.
        use_rus (bool): Apply Random Under Sampling to the training data if True.
        scale (bool): Scale the features if True.
        encode (bool): Encode categorical features if True.
        n_pca_components (int): Number of PCA components to use. 0 means no PCA.
        kbest (bool): Apply kBest feature selection if True.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
            trainloader: DataLoader for the training set.
            testloader: DataLoader for the test set.
            valloader: DataLoader for the validation set.
            dataset_sizes: Dictionary containing sizes of train, test, and val sets.
    """
    # Check if desired data engineering techniques are valid
    validate_data_args(use_smote, use_rus, n_pca_components)

    random_state = random.randint(1, 1000)

    # Load and preprocess the data
    data_path = os.path.join(CURRENT_FILE_DIR, config['data']['dataset_path'])
    data = pd.read_csv(data_path)
    data = data_cleaning(data)

    if encode:
        data = apply_encoding(data)
        log(logging.INFO, f" --> Encoding applied.")

    # Partition the training data for federated learning
    partitioner = DirichletPartitioner(
        num_partitions=n_partitions,
        partition_by="def_pay",
        alpha=10,
        min_partition_size=1000,
        self_balancing=True)
    partitioner.dataset = HFDataset.from_pandas(data, preserve_index=False)
    client_data = partitioner.load_partition(partition_id).to_pandas()

    # Split the data
    train_data, test_data, val_data = split_data(client_data, random_state)

    train_data, test_data, val_data = apply_transformations(
        train_data,
        test_data,
        val_data,
        random_state,
        **{
            "use_smote": use_smote,
            "use_rus": use_rus,
            "scale": scale,
            "encode": encode,
            "n_pca_components": n_pca_components,
            "kbest": kbest
        }
    )

    num_train, num_test, num_val = len(train_data), len(test_data), len(val_data)

    # Create custom datasets
    use_pca = True if n_pca_components > 0 else False
    trainset = DataFrameDataset(train_data, use_pca)
    testset = DataFrameDataset(test_data, use_pca)
    valset = DataFrameDataset(val_data, use_pca)

    # Create DataLoaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    log(logging.INFO, f"Client ID: {partition_id}/{n_partitions}")
    log(logging.INFO, f"Data split: (Train: {num_train} - Test: {num_test} - Val: {num_val})")
    log(logging.INFO, f"Class distribution: {train_data['def_pay'].value_counts().to_dict()}")

    dataset_sizes = {
        "trainset": num_train,
        "testset": num_test,
        "valset": num_val
    }

    return trainloader, testloader, valloader, dataset_sizes


def load_dmatrix(
        partition_id: int,
        n_partitions: int,
        batch_size: int = 32,
        use_smote: bool = False,
        use_rus: bool = False,
        scale: bool = False,
        encode: bool = False,
        n_pca_components: int = 0,
        kbest: bool = False
) -> [xgb.core.DMatrix, xgb.core.DMatrix, xgb.core.DMatrix, int, int, int]:
    """
    Load and partition the dataset for federated learning (XGBoost).

    This function:
      - Loads data from a CSV file.
      - Cleans and optionally encodes the data.
      - Partitions the data using an IID partitioner.
      - Splits the data into train/validation/test sets.
      - Optionally applies scaling, SMOTE, RUS, PCA, and kBest feature selection.
      - Converts the data into XGBoost DMatrix format.

    Args:
        partition_id (int): The partition (client) ID.
        n_partitions (int): The total number of partitions (clients).
        batch_size (int): Batch size for DataLoader (not directly used for DMatrix).
        use_smote (bool): Apply SMOTE to the training data if True.
        use_rus (bool): Apply Random Under Sampling if True.
        scale (bool): Scale the features if True.
        encode (bool): Encode categorical features if True.
        n_pca_components (int): Number of PCA components to use. 0 means no PCA.
        kbest (bool): Apply kBest feature selection if True.

    Returns:
        Tuple[xgb.core.DMatrix, xgb.core.DMatrix, xgb.core.DMatrix, int, int, int]:
            train_dmatrix: DMatrix for training set.
            test_dmatrix: DMatrix for test set.
            valid_dmatrix: DMatrix for validation set.
            num_train: Number of training samples.
            num_test: Number of test samples.
            num_val: Number of validation samples.
    """
    # Check if desired data engineering techniques are valid
    validate_data_args(use_smote, use_rus, n_pca_components)

    random_state = random.randint(1, 1000)

    # Load and preprocess the data
    data_path = os.path.join(CURRENT_FILE_DIR, config['data']['dataset_path'])
    data = pd.read_csv(data_path)
    data = data_cleaning(data)

    if encode:
        data = apply_encoding(data)
        log(logging.INFO, f" --> Encoding applied.")

    # Partition data
    partitioner = DirichletPartitioner(
        num_partitions=n_partitions,
        partition_by="def_pay",
        alpha=10,
        min_partition_size=1000,
        self_balancing=True)
    partitioner.dataset = HFDataset.from_pandas(data, preserve_index=False)
    client_data = partitioner.load_partition(partition_id).to_pandas()

    # Split the data
    train_data, test_data, val_data = split_data(client_data, random_state)

    train_data, test_data, val_data = apply_transformations(
        train_data,
        test_data,
        val_data,
        random_state,
        **{
            "use_smote": use_smote,
            "use_rus": use_rus,
            "scale": scale,
            "encode": encode,
            "n_pca_components": n_pca_components,
            "kbest": kbest
        }
    )

    num_train, num_test, num_val = len(train_data), len(test_data), len(val_data)

    log(logging.INFO, f"Client ID: {partition_id}/{n_partitions}")
    log(logging.INFO, f"Data split: (Train: {num_train} - Test: {num_test} - Val: {num_val})")
    log(logging.INFO, f"Class distribution: {train_data['def_pay'].value_counts().to_dict()}")

    train_dmatrix = transform_dataset_to_dmatrix(train_data)
    test_dmatrix = transform_dataset_to_dmatrix(test_data)
    valid_dmatrix = transform_dataset_to_dmatrix(val_data)

    return train_dmatrix, test_dmatrix, valid_dmatrix, num_train, num_test, num_val


def validate_data_args(smote, rus, pca):
    if smote and rus:
        raise ValueError("Only one of SMOTE or RUS can be applied at a time.")
    if pca != 0 and pca < 2:
        raise ValueError("n_pca_components should be at least 2 if not zero.")


def apply_transformations(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        val_data: pd.DataFrame,
        random_state: int,
        **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if kwargs.get('scale', False):
        train_data, test_data, val_data = apply_scaling(train_data, test_data, val_data)
        log(logging.INFO, f" --> Scaling applied.")

    if kwargs.get('use_smote', False):
        train_data, test_data, val_data = apply_smote(train_data, test_data, val_data, random_state)
        log(logging.INFO, f" --> SMOTE applied.")

    if kwargs.get('use_rus', False):
        train_data, test_data, val_data = apply_rus(train_data, test_data, val_data, random_state)
        log(logging.INFO, f" --> RUS applied.")

    if kwargs.get('n_pca_components', 0) > 0:
        train_data, test_data, val_data = apply_pca(train_data, test_data, val_data, kwargs['n_pca_components'])
        log(logging.INFO, f" --> PCA applied.")

    if kwargs.get('kbest', False):
        train_data, test_data, val_data = apply_kbest(
            train_data,
            test_data,
            val_data,
            ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'def_pay', 'index']
        )
        log(logging.INFO, f" --> kBest applied.")

    return train_data, test_data, val_data
