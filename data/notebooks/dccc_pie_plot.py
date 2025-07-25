import os

import matplotlib.pyplot as plt
import pandas as pd

from configs.config_loader import load_datasets_config, load_federated_config

DATASET = "taiwan"

dataset_config = load_datasets_config()[DATASET]
federated_config = load_federated_config()["ml_pipeline_experiments"]

label_name = dataset_config["target"]

CURRENT_FILE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

data_path = str(os.path.join(CURRENT_FILE_DIR, dataset_config["path"]))
df = pd.read_csv(data_path)
df.rename(columns={"default.payment.next.month": "def_pay"}, inplace=True)
df.drop('ID', axis=1, inplace=True)

# How many defaulters
perc_default = df.def_pay.sum() / len(df.def_pay)
print(f'The percentage of defaulters in the data is {perc_default * 100} %')
df['def_pay'].value_counts().plot(kind='pie', explode=[0.1, 0], autopct="%1.1f%%")
plt.plot()
plt.show()
