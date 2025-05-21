import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.config_loader import load_datasets_config, load_federated_config

DATASET = "taiwan"

dataset_config = load_datasets_config()[DATASET]
federated_config = load_federated_config()["fairness_experiments"]

label_name = dataset_config["target"]

CURRENT_FILE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

data_path = str(os.path.join(CURRENT_FILE_DIR, dataset_config["path"]))
df_master = pd.read_csv(data_path)

# 1. Ayrı hedef ve hassas öznitelikleri kaydet
Y = df_master[dataset_config["target"]]
A = df_master["SEX"]

# 2. Diğer özellikleri one-hot encode et
X_base = pd.get_dummies(
    df_master.drop(columns=[dataset_config["target"], "SEX"]),
    drop_first=True
)

df_base = pd.concat([
    X_base,
    Y.rename(dataset_config["target"]),
    A.rename("SEX")
], axis=1)

# 3. Interest sütununu ekle (ayrımcı sinyal)
X = X_base.copy()
X["Interest"] = np.random.normal(
    loc=2 * Y,
    scale=A
)

# 4. Hedef ve sensitive sütunlarını yeniden birleştir
df = pd.concat([
    X,
    Y.rename(dataset_config["target"]),
    A.rename("SEX")
], axis=1)

fig, (ax_1, ax_2) = plt.subplots(ncols=2, figsize=(10, 4), sharex=True, sharey=True)
# Men KDE
X["Interest"][(A == 1) & (Y == 0)].plot(
    kind="kde",
    label="Payment on Time",
    ax=ax_1
)
X["Interest"][(A == 1) & (Y == 1)].plot(
    kind="kde",
    label="Payment Default",
    ax=ax_1
)
ax_1.set_xlabel("Interest for Men")

# Women KDE
X["Interest"][(A == 2) & (Y == 0)].plot(
    kind="kde",
    label="Payment on Time",
    ax=ax_2
)
X["Interest"][(A == 2) & (Y == 1)].plot(
    kind="kde",
    label="Payment Default",
    ax=ax_2
)
ax_2.set_xlabel("Interest for Women")
ax_2.legend()

# Grid ve düzen
ax_1.grid(True)
ax_2.grid(True)
plt.tight_layout()
plt.savefig("synthetic_noise_dccc")
plt.show()
