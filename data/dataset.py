import logging
import os
import random
from typing import Dict, Tuple, Union

import pandas as pd
import torch
import xgboost as xgb
from datasets import DatasetDict, Dataset as HFDataset
from flwr.common.logger import log
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from configs.config import config

# Set up the data root directory
current_file_directory = os.path.dirname(os.path.abspath(__file__))


# PyTorch Dataset sınıfı
class DataFrameDataset(Dataset):
    """
    Custom PyTorch Dataset class for loading data from a DataFrame.
    """

    def __init__(self, data_frame: pd.DataFrame):
        self.data = data_frame.reset_index(drop=True)
        if config['data']['pca'] or config['data']['ica']:
            self.features = self.data.drop(columns=['def_pay'])
        else:
            self.features = self.data.drop(columns=['def_pay', 'index']).values
        self.labels = self.data['def_pay'].values
        self.indices = self.data.index.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label


def data_preprocess(data: pd.DataFrame, encode: bool = False) -> pd.DataFrame:
    """
    Preprocess the data by renaming columns, dropping unnecessary columns,
    adding an index column, and optionally scaling the features.

    Args:
        data (pd.DataFrame): The raw data.

    Returns:
        pd.DataFrame: The preprocessed data.
    """

    # Rename columns for consistency
    data = data.rename(columns={
        'default.payment.next.month': 'def_pay',
        'PAY_0': 'PAY_1'})

    # Drop the 'ID' column as it's not needed and add index column
    data = data.drop(['ID'], axis=1)
    data['index'] = data.index

    if encode:
        categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE']

        encoder = OneHotEncoder(sparse_output=False)

        X_categorical = data[categorical_columns]
        X_encoded = encoder.fit_transform(X_categorical)

        encoded_feature_names = encoder.get_feature_names_out(categorical_columns)
        df_encoded = pd.DataFrame(X_encoded, columns=encoded_feature_names)
        df_numeric = data.drop(columns=categorical_columns)
        data = pd.concat([df_numeric.reset_index(drop=True),
                          df_encoded.reset_index(drop=True)], axis=1)

    return data


def split_data(data: pd.DataFrame, smote: bool = False, scale: bool = False, pca: bool = False, ica: bool = False,
               rus: bool = False) -> \
        Tuple[
            pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training, validation, and test sets.
    Optionally apply SMOTE to the training data to address class imbalance.

    Args:
        data (pd.DataFrame): The preprocessed data.
        smote (bool): Whether to apply SMOTE to the training data.
        scale (bool): Whether to apply standard scaling to the features.
        pca (bool): Whether to apply PCA to the features.
        ica (bool): Whether to apply ICA to the features.
        rus (bool): Whether to apply RUS to the training data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The training, test, and validation datasets.
    """
    assert not (pca and ica), "Apply only PCA or ICA at a time."
    assert not (rus and smote), "Apply only RUS or SMOTE at a time."

    random_state = random.randint(1, 1000)
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=random_state)
    test_data, val_data = train_test_split(temp_data, test_size=1 / 3, random_state=random_state)

    test_data = test_data[train_data.columns]
    val_data = val_data[train_data.columns]

    if scale:
        # Apply standard scaling to the features
        cols_to_exclude = train_data.columns[train_data.columns.str.contains('SEX|EDUCATION|MARRIAGE')].tolist()
        cols_to_exclude.extend(['def_pay', 'index'])
        feature_columns = [col for col in train_data.columns if col not in cols_to_exclude]

        scaler = StandardScaler()

        # Fıt and transform on Training data
        scaled_training_data = scaler.fit_transform(train_data[feature_columns])
        scaled_training_data = pd.DataFrame(scaled_training_data, columns=feature_columns, index=train_data.index)

        # Only transform on Test data
        scaled_test_data = scaler.transform(test_data[feature_columns])
        scaled_test_data = pd.DataFrame(scaled_test_data, columns=feature_columns, index=test_data.index)

        # Only transform on Validation data
        scaled_val_data = scaler.transform(val_data[feature_columns])
        scaled_val_data = pd.DataFrame(scaled_val_data, columns=feature_columns, index=val_data.index)

        train_data[feature_columns] = scaled_training_data
        test_data[feature_columns] = scaled_test_data
        val_data[feature_columns] = scaled_val_data

        test_data = test_data[train_data.columns]
        val_data = val_data[train_data.columns]

    if smote:
        # Apply SMOTE to the training data
        smote_processor = SMOTE(random_state=random_state)
        X_train = train_data.drop(columns=['def_pay', 'index'])
        y_train = train_data['def_pay']
        X_resampled, y_resampled = smote_processor.fit_resample(X_train, y_train)

        # Reconstruct the training DataFrame
        train_data = pd.DataFrame(X_resampled, columns=X_train.columns)
        train_data['def_pay'] = y_resampled
        train_data = train_data.reset_index(drop=True)
        train_data['index'] = train_data.index

        print(f"Class distribution after applying SMOTE: {train_data['def_pay'].value_counts()}")

        test_data = test_data[train_data.columns]
        val_data = val_data[train_data.columns]

    if rus:
        rus = RandomUnderSampler(random_state=random_state)
        X_train = train_data.drop(columns=['def_pay', 'index'])
        y_train = train_data['def_pay']
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

        # Reconstruct the training DataFrame
        train_data = pd.DataFrame(X_resampled, columns=X_train.columns)
        train_data['def_pay'] = y_resampled
        train_data = train_data.reset_index(drop=True)
        train_data['index'] = train_data.index

        print(f"Class distribution after applying RUS: {train_data['def_pay'].value_counts()}")

        test_data = test_data[train_data.columns]
        val_data = val_data[train_data.columns]

    if pca:
        log(
            logging.WARNING,
            f"\nPCA: {pca}"
        )
        pca_model = PCA(n_components=pca)

        X_train = train_data.drop(columns=['def_pay', 'index']).reset_index(drop=True)
        y_train = train_data['def_pay'].reset_index(drop=True)
        train_data = pd.DataFrame(pca_model.fit_transform(X_train))
        train_data['def_pay'] = y_train

        X_test = test_data.drop(columns=['def_pay', 'index']).reset_index(drop=True)
        y_test = test_data['def_pay'].reset_index(drop=True)
        test_data = pd.DataFrame(pca_model.transform(X_test))
        test_data['def_pay'] = y_test

        X_val = val_data.drop(columns=['def_pay', 'index']).reset_index(drop=True)
        y_val = val_data['def_pay'].reset_index(drop=True)
        val_data = pd.DataFrame(pca_model.transform(X_val))
        val_data['def_pay'] = y_val

        log(
            logging.WARNING,
            f"\nAfter PCA shape is: {train_data.shape[1]}"
        )

        test_data = test_data[train_data.columns]
        val_data = val_data[train_data.columns]

    if ica:
        log(
            logging.WARNING,
            f"\nPCA: {ica}"
        )
        ica_model = FastICA(n_components=ica)

        X_train = train_data.drop(columns=['def_pay', 'index']).reset_index(drop=True)
        y_train = train_data['def_pay'].reset_index(drop=True)
        train_data = pd.DataFrame(ica_model.fit_transform(X_train))
        train_data['def_pay'] = y_train

        X_test = test_data.drop(columns=['def_pay', 'index']).reset_index(drop=True)
        y_test = test_data['def_pay'].reset_index(drop=True)
        test_data = pd.DataFrame(ica_model.transform(X_test))
        test_data['def_pay'] = y_test

        X_val = val_data.drop(columns=['def_pay', 'index']).reset_index(drop=True)
        y_val = val_data['def_pay'].reset_index(drop=True)
        val_data = pd.DataFrame(ica_model.transform(X_val))
        val_data['def_pay'] = y_val

        log(
            logging.WARNING,
            f"\nAfter ICA shape is: {train_data.shape[1]}"
        )

        test_data = test_data[train_data.columns]
        val_data = val_data[train_data.columns]

    return train_data, test_data, val_data


def transform_dataset_to_dmatrix(data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
    """Transform dataset to DMatrix format for xgboost."""
    y = data["def_pay"]
    x = data.drop(columns=['def_pay', 'index'])
    new_data = xgb.DMatrix(x, label=y)
    return new_data


def load_data(
        partition_id: int,
        n_partitions: int,
        batch_size: int = 32,
        smote: bool = False,
        rus: bool = False,
        scale: bool = False,
        encode: bool = False,
        pca: bool = False,
        ica: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Load the dataset and partition it for federated learning.
    Returns DataLoaders for training, validation, and test sets.

    Args:
        partition_id (int): The partition ID for the client.
        n_partitions (int): Total number of partitions (clients).
        batch_size (int): Batch size for DataLoader.
        smote (bool): Whether to apply SMOTE to the training data.
        rus (bool): Whether to apply RUS to the training data.
        scale (bool): Whether to scale the features.
        encode (bool): Whether to encode the features.
        pca (bool): Whether to PCA the features.
        ica (bool): Whether to ICA the features.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]: DataLoaders and dataset sizes.
    """

    # Load and preprocess the data
    data = pd.read_csv(current_file_directory + config['data']['dataset_path'])
    data = data_preprocess(data, encode=encode)

    # Partition the training data for federated learning
    partitioner = DirichletPartitioner(num_partitions=n_partitions,
                                       partition_by="def_pay",
                                       alpha=10,
                                       min_partition_size=1000,
                                       self_balancing=True)
    partitioner.dataset = HFDataset.from_pandas(data, preserve_index=False)
    client_data = partitioner.load_partition(partition_id).to_pandas()

    # Split the data
    train_data, test_data, val_data = split_data(client_data, scale=scale, smote=smote, pca=pca, ica=ica, rus=rus)

    # train_data = train_data[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'def_pay', 'index']]
    # test_data = test_data[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'def_pay', 'index']]
    # val_data = val_data[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'def_pay', 'index']]

    num_train = len(train_data)
    num_test = len(test_data)
    num_val = len(val_data)

    # Create custom datasets
    trainset = DataFrameDataset(train_data)
    testset = DataFrameDataset(test_data)
    valset = DataFrameDataset(val_data)

    # Create DataLoaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    log(
        logging.WARNING,
        f"\nClient ID: {partition_id}/{n_partitions} "
        f"\nData split: (Train - Test - Val) {num_train} - {num_test} - {num_val} "
        f"\nClass distribution: \n{train_data['def_pay'].value_counts()} "
    )

    dataset_sizes = {
        "trainset": num_train,
        "testset": num_test,
        "valset": num_val
    }

    return trainloader, testloader, valloader, dataset_sizes


def load_xgboost_data(
        partition_id: int,
        n_partitions: int,
        batch_size: int = 32,
        rus: bool = False,
        smote: bool = False,
        scale: bool = False,
        encode: bool = False,
        pca: bool = False,
        ica: bool = False
) -> [xgb.core.DMatrix, xgb.core.DMatrix, xgb.core.DMatrix, int, int, int]:
    """
    Load the dataset and partition it for federated learning XGBoost.
    Returns DataLoaDMatrixders for training, validation, and test sets. Also the length of them

    Args:
        partition_id (int): The partition ID for the client.
        n_partitions (int): Total number of partitions (clients).
        batch_size (int): Batch size for DataLoader.
        smote (bool): Whether to apply SMOTE to the training data.
        scale (bool): Whether to scale the features.
        encode (bool): Whether to encode the features.
        pca (bool): Whether to PCA the features.
        ica (bool): Whether to ICA the features.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]: DataLoaders and dataset sizes.
    """
    # Load and preprocess the data
    data = pd.read_csv(current_file_directory + config['data']['dataset_path'])
    data = data_preprocess(data, encode=encode)

    # Partition data
    partitioner = IidPartitioner(num_partitions=n_partitions)
    partitioner.dataset = HFDataset.from_pandas(data, preserve_index=False)
    client_data = partitioner.load_partition(partition_id).to_pandas()

    # Split the data
    train_data, test_data, val_data = split_data(client_data, scale=scale, smote=smote, pca=pca, ica=ica, rus=rus)

    train_data = train_data[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'def_pay', 'index']]
    test_data = test_data[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'def_pay', 'index']]
    val_data = val_data[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'def_pay', 'index']]

    num_train = len(train_data)
    num_test = len(test_data)
    num_val = len(val_data)

    log(
        logging.WARNING,
        f"\nClient ID: {partition_id}/{n_partitions} "
        f"\nData split: (Train - Test - Val) {num_train} - {num_test} - {num_val} "
        f"\nClass distribution: \n{train_data['def_pay'].value_counts()} "
    )

    train_dmatrix = transform_dataset_to_dmatrix(train_data)
    test_dmatrix = transform_dataset_to_dmatrix(test_data)
    valid_dmatrix = transform_dataset_to_dmatrix(val_data)

    return train_dmatrix, test_dmatrix, valid_dmatrix, num_train, num_test, num_val
