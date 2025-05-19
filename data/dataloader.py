import gc
import logging
import os
import random
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
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
    PROPORTIONS = [
        {1: 0.20, 2: 0.80},  # client0
        {1: 0.30, 2: 0.70},  # client1
        {1: 0.40, 2: 0.60},  # client2
        {1: 0.50, 2: 0.50},  # client3
        {1: 0.75, 2: 0.25},  # client4
    ]

    @staticmethod
    def create_federated_loaders(
            dataset_config: dict,
            federated_config: dict,
            partition_id: int,
            num_clients: int,
            load_dmatrix=False
    ) -> tuple[
        Union[DataLoader, xgb.core.DMatrix],
        Union[DataLoader, xgb.core.DMatrix],
        Union[DataLoader, xgb.core.DMatrix],
        Union[dict, tuple[int, int, int]]
    ]:
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

        if 'drop_columns' in dataset_config:
            df = df.drop(columns=dataset_config['drop_columns'])

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

        if load_dmatrix:
            num_train, num_test, num_val = len(train_df), len(test_df), len(val_df)

            y = train_df[dataset_config["target"]]
            x = train_df.drop(columns=dataset_config["target"])
            train_dmatrix = xgb.DMatrix(x, label=y)

            y = test_df[dataset_config["target"]]
            x = test_df.drop(columns=dataset_config["target"])
            test_dmatrix = xgb.DMatrix(x, label=y)

            y = val_df[dataset_config["target"]]
            x = val_df.drop(columns=dataset_config["target"])
            valid_dmatrix = xgb.DMatrix(x, label=y)

            train_loader, test_loader, val_loader = train_dmatrix, test_dmatrix, valid_dmatrix
            sizes = (num_train, num_test, num_val)
        else:
            # 8) PyTorch Dataset & DataLoader
            use_pca = dataset_config.get("pca", 0) > 0
            train_set = DataFrameDataset(train_df, use_pca, dataset_config["target"])
            test_set = DataFrameDataset(test_df, use_pca, dataset_config["target"])
            val_set = DataFrameDataset(val_df, use_pca, dataset_config["target"])

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

    @staticmethod
    def create_federated_unfair_loaders(
            dataset_config: dict,
            federated_config: dict,
            partition_id: int,
            num_clients: int,
            **kwargs
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

        # 1) Ham veriyi yükle
        data_path = str(os.path.join(CURRENT_FILE_DIR, dataset_config["path"]))
        df_master = pd.read_csv(data_path)
        log(logging.INFO, f"[DataLoader] Loaded {dataset_config['path']} shape={df_master.shape}")

        # 1. Ayrı hedef ve hassas öznitelikleri kaydet
        Y_master = df_master[dataset_config["target"]]
        A_master = df_master["SEX"]

        # 2. Diğer özellikleri one-hot encode et
        X_base = pd.get_dummies(
            df_master.drop(columns=[dataset_config["target"], "SEX"]),
            drop_first=True
        )

        NUM_SPLITS = 5
        BASE_SEED = 42

        df_base = pd.concat([
            X_base,
            Y_master.rename(dataset_config["target"]),
            A_master.rename("SEX")
        ], axis=1)

        client_idxs = custom_split_by_sex(df_base, NUM_SPLITS, BASE_SEED, DataLoaderFactory.PROPORTIONS)

        print("=== Client bazında yeni SEX dağılımları ===")
        for idx, idxs in enumerate(client_idxs):
            sub = df_base.loc[idxs, "SEX"]
            counts = sub.value_counts()
            ratios = sub.value_counts(normalize=True)
            print(f"Client {idx}:")
            # SEX=1 önce, sonra SEX=2 olacak şekilde sıralama
            for sex_val in sorted(counts.index):
                print(f"  SEX={sex_val}: {counts[sex_val]} kişi ({ratios[sex_val]:.2%})")
        print()

        # 3. Interest sütununu ekle (ayrımcı sinyal)
        X = X_base.copy()
        X["Interest"] = np.random.normal(
            loc=2 * Y_master,
            scale=A_master
        )

        # 4. Hedef ve sensitive sütunlarını yeniden birleştir
        df = pd.concat([
            X,
            Y_master.rename(dataset_config["target"]),
            A_master.rename("SEX")
        ], axis=1)

        client_dfs = client_idxs

        logging.info("[DataLoader] Interest feature injected for fairness experiments on Taiwan dataset")

        # 2) Encode
        if dataset_config["encode"]:
            df = apply_encoding(df)
            log(logging.INFO, " → Encoding applied")

        idxs = client_dfs[partition_id]
        client_df = df.loc[idxs]

        seed_split = BASE_SEED + np.random.randint(0, 100)
        train_df, test_df, val_df = split_data(client_df, seed_split)

        # 6) Dönüşümler (SMOTE / RUS / PCA / kBest)
        train_df, test_df, val_df = apply_transformations(
            train_df,
            test_df,
            val_df,
            BASE_SEED,
            use_smote=dataset_config["smote"],
            use_rus=dataset_config["rus"],
            scale=dataset_config["scale"],
            encode=dataset_config["encode"],
            n_pca_components=dataset_config["pca"],
            kbest=dataset_config["kbest"]
        )

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
        algorithm_name: str,
        dataset_config: dict,
        federated_config: dict,
        **kwargs
):
    if algorithm_name == "xgboost":
        return DataLoaderFactory.create_federated_loaders(
            dataset_config=dataset_config,
            federated_config=federated_config,
            partition_id=partition_id,
            num_clients=num_clients,
            load_dmatrix=True
        )

    elif federated_config.get("experiment_name") == "fairness_experiments" and dataset_config["name"] == "taiwan":
        return DataLoaderFactory.create_federated_unfair_loaders(
            dataset_config=dataset_config,
            federated_config=federated_config,
            partition_id=partition_id,
            num_clients=num_clients,
        )

    else:
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


def custom_split_by_sex(df: pd.DataFrame,
                        n_splits: int,
                        seed: int,
                        proportions: list[dict[int, float]]):
    """
    Her bir client için sabit SEX oranları ile, eşit büyüklükte parçalar oluşturur.
    proportions: [{sex_val: oran, ...}, ...] uzunluğu n_splits
    """
    rs = np.random.RandomState(seed)
    N = len(df)
    base = N // n_splits
    rem = N % n_splits
    # client boyutları
    sizes = [base + (1 if i < rem else 0) for i in range(n_splits)]
    # grup bazında karışık indeks havuzları
    pools = {}
    for sex in df["SEX"].unique():
        idxs = df.index[df["SEX"] == sex].to_list()
        rs.shuffle(idxs)
        pools[sex] = idxs

    client_indices = []
    for i in range(n_splits):
        ci = []
        size = sizes[i]
        prop = proportions[i]
        # her sex için hedef adet
        targets = {sex: int(round(prop.get(sex, 0) * size)) for sex in pools}
        # toplamdan sapma varsa en yüksek orana ekle
        diff = size - sum(targets.values())
        if diff:
            # en büyük prop key’si
            major = max(prop, key=prop.get)
            targets[major] += diff
        # havuzdan al
        for sex, cnt in targets.items():
            take = pools[sex][:cnt]
            ci.extend(take)
            pools[sex] = pools[sex][cnt:]
        rs.shuffle(ci)
        client_indices.append(ci)
    return client_indices
