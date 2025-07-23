import os

import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset as HFDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions

from configs.config_loader import load_datasets_config, load_federated_config

DATASET = "taiwan"

dataset_config = load_datasets_config()[DATASET]
federated_config = load_federated_config()["ml_pipeline_experiments"]

label_name = dataset_config["target"]

CURRENT_FILE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

data_path = str(os.path.join(CURRENT_FILE_DIR, dataset_config["path"]))
df = pd.read_csv(data_path)

partitioner = DirichletPartitioner(
    num_partitions=5,
    partition_by=dataset_config["target"],
    alpha=2,
    self_balancing=True,
)
partitioner.dataset = HFDataset.from_pandas(df, preserve_index=False)

plt.rcParams.update({
    "font.size": 16,           # genel font
    "axes.titlesize": 16,      # başlık
    "axes.labelsize": 14,      # eksen etiketleri
    "xtick.labelsize": 12,     # x-tick etiketleri
    "ytick.labelsize": 12,     # y-tick etiketleri
    "legend.fontsize": 12,     # legend (zaten kaldırdıysan gerek kalmaz)
})


# Bar plot
fig, ax, dist_df = plot_label_distributions(
    partitioner=partitioner,
    label_name=label_name,
    plot_type="bar",
    size_unit="absolute",
    legend=False,
    verbose_labels=True,
    figsize=(8, 6),
    title=f"Label Distribution per Client for {DATASET} Dataset"
)
fig.tight_layout()
plt.savefig(f"{DATASET}_DirichletPartitioner_bar")

# Heatmap ile
fig, ax, dist_df = plot_label_distributions(
    partitioner=partitioner,
    label_name=label_name,
    plot_type="heatmap",
    size_unit="percent",
    plot_kwargs={"annot": True},
    legend=True,
    verbose_labels=True,
    figsize=(8, 6),
    title=f"Label Percentages per Client for {DATASET} Dataset"
)
fig.tight_layout()
plt.savefig(f"{DATASET}_DirichletPartitioner_heatmap")
