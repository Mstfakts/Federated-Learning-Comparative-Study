import gc
import logging
import os
import random
from typing import Tuple

import pandas as pd
import torch
from datasets import Dataset as HFDataset
from flwr.common.logger import log
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader, Dataset

from data.data_process import (
    split_data,
    apply_smote,
    apply_rus,
    apply_pca,
    apply_scaling,
    apply_kbest,
    apply_encoding,
)

# Set up the data root directory
CURRENT_FILE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DataLoaderFactory:
    @staticmethod
    def create_federated_loaders(
            dataset_config: dict,
            federated_config: dict,
            partition_id: int,
            num_clients: int,
    ) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
        """
        1) CSV'i oku
        2) (Opsiyonel) encode
        3) Dirichlet partitioner ile böl
        4) chunk → DataFrame, birleştir
        5) train/test/val split
        6) apply_transformations (SMOTE/RUS/PCA/kBest)
        7) PyTorch Dataset & DataLoader
        8) Son olarak boyutları return et
        """

        # 0) Validate dönüşümler
        validate_data_args(
            dataset_config["smote"],
            dataset_config["rus"],
            dataset_config["pca"],
        )
        rnd = random.randint(1, 1_000)

        # 1) Ham veriyi yükle
        data_path = str(os.path.join(CURRENT_FILE_DIR, dataset_config["path"]))
        df = pd.read_csv(data_path)
        log(logging.INFO, f"[DataLoader] Loaded {dataset_config['path']} shape={df.shape}")

        # 2) Encode
        if dataset_config["encode"]:
            df = apply_encoding(df)
            log(logging.INFO, " → Encoding applied")

        # 3) Federated partition
        partitioner = DirichletPartitioner(
            num_partitions=num_clients,
            partition_by=dataset_config["target"],
            alpha=federated_config["partitioner"]["alpha"],
            min_partition_size=federated_config["partitioner"]["min_partition_size"],
            self_balancing=True,
        )
        partitioner.dataset = HFDataset.from_pandas(df, preserve_index=False)
        del df
        gc.collect()

        # 4) chunk → DataFrame birleştirme
        client_ds = partitioner.load_partition(partition_id)
        chunks = []
        for chunk in load_partition_in_chunks(
                client_ds, chunk_size=federated_config["partitioner"]["chunk_size"]
        ):
            chunks.append(chunk)
        client_df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()

        # 5) Train/Test/Val split
        train_df, test_df, val_df = split_data(
            client_df,
            rnd
        )
        del client_df
        gc.collect()

        # 6) Dönüşümler (SMOTE / RUS / PCA / kBest)
        train_df, test_df, val_df = apply_transformations(
            train_df,
            test_df,
            val_df,
            rnd,
            use_smote=dataset_config["smote"],
            use_rus=dataset_config["rus"],
            scale=dataset_config["scale"],
            encode=dataset_config["encode"],
            n_pca_components=dataset_config["pca"],
            kbest=dataset_config["kbest"]
        )

        # 7) ID sütunu varsa drop
        # TODO bus adece 1 veriseti için geçerli ona göre genel bir düzenleme yapılmalı
        for df_ in (train_df, test_df, val_df):
            if "SK_ID_CURR" in df_.columns:
                df_.drop(columns=["SK_ID_CURR"], inplace=True)

        # 8) PyTorch Dataset & DataLoader
        use_pca = dataset_config.get("pca", 0) > 0
        train_set = DataFrameDataset(train_df, use_pca, dataset_config["target"])
        test_set = DataFrameDataset(test_df, use_pca, dataset_config["target"])
        val_set = DataFrameDataset(val_df, use_pca, dataset_config["target"])

        # TODO XGBoost için düzenleme lazım
        batch_size = dataset_config["batch_size"]
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        sizes = {
            "train": len(train_df),
            "test": len(test_df),
            "val": len(val_df),
        }
        log(logging.INFO, f"[DataLoader] Partition {partition_id}/{num_clients} sizes: {sizes}")

        return train_loader, test_loader, val_loader, sizes


def partition_data_loader(
        partition_id: int,
        num_clients: int,
        dataset_config: dict,
        federated_config: dict,
):
    return DataLoaderFactory.create_federated_loaders(
        dataset_config=dataset_config,
        federated_config=federated_config,
        partition_id=partition_id,
        num_clients=num_clients,
    )


class DataFrameDataset(Dataset):
    """
    Custom PyTorch Dataset class for loading features and labels from a DataFrame.

    Args:
        data_frame (pd.DataFrame): The DataFrame containing the data.
        use_pca (bool): Whether PCA was applied or not. If True, 'index' column may be omitted.
    """

    def __init__(self, data_frame: pd.DataFrame, use_pca: bool = False, target: str = ""):
        self.data = data_frame.reset_index(drop=True)
        self.labels = self.data[target].values

        # Determine which columns to drop
        drop_columns = [target]
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


def load_partition_in_chunks(hf_dataset, chunk_size=50_000):
    """
    Hugging Face Dataset'i chunk_size boyutunda parçalara bölerek
    her birini pandas DataFrame olarak yield eden jeneratör.
    """
    length = len(hf_dataset)
    for start_idx in range(0, length, chunk_size):
        end_idx = min(start_idx + chunk_size, length)
        # HF dataset'in bir parçasını seç
        subset = hf_dataset.select(range(start_idx, end_idx))
        # O parçayı pandas'a dönüştür
        df_chunk = subset.to_pandas()
        yield df_chunk


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
        log(logging.INFO, f" --> Scaling applied. Shape: {train_data.shape}")

    if kwargs.get('use_smote', False):
        train_data, test_data, val_data = apply_smote(train_data, test_data, val_data, random_state)
        log(logging.INFO, f" --> SMOTE applied. Shape: {train_data.shape}")

    if kwargs.get('use_rus', False):
        train_data, test_data, val_data = apply_rus(train_data, test_data, val_data, random_state)
        log(logging.INFO, f" --> RUS applied. Shape: {train_data.shape}")

    if kwargs.get('n_pca_components', 0) > 0:
        train_data, test_data, val_data = apply_pca(train_data, test_data, val_data, kwargs['n_pca_components'])
        log(logging.INFO, f" --> PCA applied. Shape: {train_data.shape}")

    if kwargs.get('kbest', False):
        train_data, test_data, val_data = apply_kbest(
            train_data,
            test_data,
            val_data,
            ['DAYS_BIRTH', 'DAYS_ID_PUBLISH', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'EXT_SOURCE_1',
             'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_LAST_PHONE_CHANGE', 'CODE_GENDER_F', 'CODE_GENDER_M',
             'NAME_INCOME_TYPE_Working', 'NAME_EDUCATION_TYPE_Higher education',
             'NAME_EDUCATION_TYPE_Secondary / secondary special', 'preapp_CODE_REJECT_REASON_HC_mean',
             'preapp_NAME_PRODUCT_TYPE_walk-in_mean', 'preapp_NAME_CONTRACT_STATUS_Refused_mean',
             'preapp_NFLAG_INSURED_ON_APPROVAL_Missing_mean', 'preapp_NAME_CONTRACT_STATUS_Approved_mean',
             'preapp_CODE_REJECT_REASON_XAP_mean', 'preapp_DAYS_DECISION_min', 'preapp_DAYS_FIRST_DRAWING_count',
             'preapp_DAYS_FIRST_DUE_min', 'preapp_DAYS_LAST_DUE_1ST_VERSION_min', 'preapp_DAYS_LAST_DUE_min',
             'client_installments_DAYS_ENTRY_PAYMENT_min_min', 'client_installments_DAYS_INSTALMENT_min_min',
             'client_installments_DAYS_INSTALMENT_mean_min', 'client_installments_DAYS_ENTRY_PAYMENT_mean_min',
             'client_installments_DAYS_ENTRY_PAYMENT_max_min', 'client_installments_DAYS_INSTALMENT_max_min', 'TARGET']
        )
        log(logging.INFO, f" --> kBest applied. Shape: {train_data.shape}")

    return train_data, test_data, val_data
