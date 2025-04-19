import os

os.environ["config_file"] = "linear_svc"
from configs.config import get_config
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)
import warnings

warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated")
import numpy as np

CURRENT_FILE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore', message='\'n_jobs\' > 1 does not have any effect')
np.random.seed(42)
config = get_config()
config['client'] = 1

data_path = str(os.path.join(CURRENT_FILE_DIR, config['data']['dataset_path']))
data = pd.read_csv(data_path)

num_cols = data.select_dtypes(include=['float64', 'int64'])
cat_cols = data.select_dtypes(include=['object'])

print("Numeric Variables:")
print(num_cols.columns.tolist())

print("\nCategorical Variables:")
print(cat_cols.columns.tolist())

print("\nMissing Values:")
print(data.isnull().sum())

# Since it is continuous and right-skewed, I will fill it with Median
data['MORTDUE'] = data['MORTDUE'].fillna(data['MORTDUE'].median())
data['VALUE'] = data['VALUE'].fillna(data['VALUE'].median())
data['YOJ'] = data['YOJ'].fillna(data['YOJ'].median())
data['CLAGE'] = data['CLAGE'].fillna(data['CLAGE'].median())
data['DEBTINC'] = data['DEBTINC'].fillna(data['DEBTINC'].median())

data['DEROG'] = data['DEROG'].fillna(data['DEROG'].mode()[0])
data['DELINQ'] = data['DELINQ'].fillna(data['DELINQ'].mode()[0])
data['NINQ'] = data['NINQ'].fillna(data['NINQ'].mode()[0])
data['CLNO'] = data['CLNO'].fillna(data['CLNO'].mode()[0])

# Discrete
data['REASON'] = data['REASON'].fillna(data['REASON'].mode()[0])
data['JOB'] = data['JOB'].fillna(data['JOB'].mode()[0])

print("\nMissing Values after imputing:")
print(data.isnull().sum())

# ENCODING
encode_columns = ["REASON", "JOB"]
data = pd.get_dummies(data, columns=encode_columns, drop_first=True)
bool_columns = [col for col in data.columns if data[col].dtype == bool]
data[bool_columns] = data[bool_columns].astype(int)

# SCALING
scaler = MinMaxScaler()
data['LOAN'] = scaler.fit_transform(data[['LOAN']])
data['MORTDUE'] = scaler.fit_transform(data[['MORTDUE']])
data['VALUE'] = scaler.fit_transform(data[['VALUE']])
data['YOJ'] = scaler.fit_transform(data[['YOJ']])
data['DEROG'] = scaler.fit_transform(data[['DEROG']])
data['DELINQ'] = scaler.fit_transform(data[['DELINQ']])
data['CLAGE'] = scaler.fit_transform(data[['CLAGE']])
data['NINQ'] = scaler.fit_transform(data[['NINQ']])
data['CLNO'] = scaler.fit_transform(data[['CLNO']])
data['DEBTINC'] = scaler.fit_transform(data[['DEBTINC']])

data.to_csv(CURRENT_FILE_DIR + '/hmeq_imputed_scaled_encoded.csv', index=False)
