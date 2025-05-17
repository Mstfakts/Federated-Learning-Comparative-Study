import os
import warnings

import numpy as np
import pandas as pd
from fairlearn.metrics import equal_opportunity_difference
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
from configs.config_loader import load_datasets_config, load_algorithms_config, load_federated_config
from data.data_process import split_data
from data.dataloader import custom_split_by_sex
from data.dataloader import DataLoaderFactory

# --- Config & Veri Yükleme ---
algo_cfg = load_algorithms_config()["logistic_regression"]
ds_cfg = load_datasets_config()["taiwan"]
fdr_cfg = load_federated_config()["fairness_experiments"]

CURRENT_FILE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
    )
)
data_path = os.path.join(CURRENT_FILE_DIR, ds_cfg["path"])
df_master = pd.read_csv(data_path)

Y_master = df_master[ds_cfg["target"]]
A_master = df_master["SEX"]
X_base = pd.get_dummies(
    df_master.drop(columns=[ds_cfg["target"], "SEX"]),
    drop_first=True
)

NUM_RUNS = 100
NUM_SPLITS = 5
BASE_SEED = 42

# --- 1) Bir kez böl ve SEX dağılımını yazdır ---
# df_base index korur
df_base = pd.concat([
    X_base,
    Y_master.rename(ds_cfg["target"]),
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

# --- 2) 100 run boyunca EOD topla ---
eod_results = np.zeros((NUM_RUNS, NUM_SPLITS))

for run in range(NUM_RUNS):
    seed = BASE_SEED + run
    np.random.seed(seed)

    # Interest sütununu run başına oluştur
    X = X_base.copy()
    X["Interest"] = np.random.normal(
        loc=2 * Y_master,
        scale=A_master
    )
    df = pd.concat([
        X,
        Y_master.rename(ds_cfg["target"]),
        A_master.rename("SEX")
    ], axis=1)

    client_dfs = client_idxs

    for i, idxs in enumerate(client_dfs):
        client_df = df.loc[idxs]
        train_df, test_df, val_df = split_data(client_df, seed + i)

        # Sadece o client’ın kendi train setiyle modeli eğit
        train_y = train_df[ds_cfg["target"]]
        train_x = train_df.drop(columns=["ID", ds_cfg["target"]])
        model = LogisticRegression(**algo_cfg)
        model.fit(train_x, train_y)

        # Validation’da fairness ölçümü
        val_y = val_df[ds_cfg["target"]]
        val_x = val_df.drop(columns=["ID", ds_cfg["target"]])
        y_pred = model.predict(val_x)
        eod_results[run, i] = equal_opportunity_difference(
            y_true=val_y,
            y_pred=y_pred,
            sensitive_features=val_x["SEX"].to_numpy(),
            method="between_groups"
        )

# --- 3) Sonuçları özetle ---
avg_eod_per_client = eod_results.mean(axis=0)
overall_avg_eod = eod_results.mean()

print("=== Ortalama EOD değerleri ===")
for idx, avg in enumerate(avg_eod_per_client):
    print(f"Client {idx} EOD (ortalama over {NUM_RUNS} runs): {avg:.4f}")
print(f"Overall average EOD: {overall_avg_eod:.4f}")
