import logging
import os
import random
from typing import Dict, Tuple, Union
os.environ["config_file"] = "mlp"
import pandas as pd
from data.dataset import load_data
import numpy as np
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
import matplotlib.pyplot as plt
from configs.config import config
import seaborn as sns
# Set up the data root directory
current_file_directory = os.path.dirname(os.path.abspath(__file__))

data = pd.read_csv(current_file_directory + config['data']['dataset_path'])

train_loader, test_loader, val_loader, num_examples = load_data(
        partition_id=0,
        n_partitions=1,
        batch_size=1,
        scale=True,
        smote=False,
        rus=False,
        encode=True,
        pca=False,
        ica=False
    )


continuous_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns


from sklearn.feature_selection import SelectKBest, f_classif
selector_continuous = SelectKBest(f_classif, k=5)
X_continuous_selected = selector_continuous.fit_transform(X[continuous_features], y)

# Sürekli özellikler için skorları alma
f_classif_scores = selector_continuous.scores_

# Skorları ve sürekli özellik isimlerini bir DataFrame'de birleştirme
continuous_scores_df = pd.DataFrame({
    'Feature': continuous_features,
    'Score': f_classif_scores
}).sort_values(by='Score', ascending=False)

print("Sürekli özelliklerin etkisi:")
print(continuous_scores_df)


from sklearn.feature_selection import SelectKBest, chi2
selector_categorical = SelectKBest(chi2, k=5)
X_categorical_selected = selector_categorical.fit_transform(X_categorical_encoded, y)
# Kategorik özellikler için skorları alma
chi2_scores = selector_categorical.scores_

# One-Hot Encoding sonrası özellik isimlerini alma
categorical_feature_names = encoder.get_feature_names_out(categorical_features)

# Skorları ve kategorik özellik isimlerini bir DataFrame'de birleştirme
categorical_scores_df = pd.DataFrame({
    'Feature': categorical_feature_names,
    'Score': chi2_scores
}).sort_values(by='Score', ascending=False)

print("Kategorik özelliklerin etkisi:")
print(categorical_scores_df)


# Sürekli ve kategorik özelliklerin etkilerini birleştirme
all_features_df = pd.concat([continuous_scores_df, categorical_scores_df])

# Skorları büyükten küçüğe sıralama
all_features_df_sorted = all_features_df.sort_values(by='Score', ascending=False)

print("Tüm özelliklerin hedef değişken üzerindeki etkisi:")
print(all_features_df_sorted)

X_selected = np.hstack((X_continuous_selected, X_categorical_selected))










tmp = pd.DataFrame({'Feature': predictors_f, 'Feature importance': clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (16,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()