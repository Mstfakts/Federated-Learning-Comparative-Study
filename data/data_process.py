import logging
from typing import Tuple

import pandas as pd
import xgboost as xgb
from flwr.common.logger import log
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def split_data(data: pd.DataFrame, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training, validation, and test sets.
    Optionally apply SMOTE to the training data to address class imbalance.

    Args:
        data (pd.DataFrame): The preprocessed data.
        random_state (int):

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The training, test, and validation datasets.
    """
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=random_state)  # %70 (train) - %30 (temp)
    test_data, val_data = train_test_split(temp_data, test_size=1 / 3, random_state=random_state)  # %20 - %10

    test_data = test_data[train_data.columns]  # Make sure feature order is fix.
    val_data = val_data[train_data.columns]  # Make sure feature order is fix.

    return train_data, test_data, val_data


def apply_encoding(data):
    le = LabelEncoder()

    df_cat = data.select_dtypes(include=['object', 'bool'])
    for col in df_cat:
        if df_cat[col].dtype in ['object', 'bool']:
            # If 2 or fewer unique categories
            if len(list(df_cat[col].unique())) <= 2:
                # Train on the training data
                le.fit(df_cat[col])
                # Transform both training and testing data
                df_cat[col] = le.transform(df_cat[col])

    df_cat = pd.get_dummies(df_cat)

    cat_cols = df_cat.columns
    data[cat_cols] = df_cat

    return data


def apply_smote(train_data, test_data, val_data, random_state):
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

    log(
        logging.WARNING,
        f"Class distribution after applying SMOTE: {train_data['def_pay'].value_counts()}"
    )

    test_data = test_data[train_data.columns]
    val_data = val_data[train_data.columns]

    return train_data, test_data, val_data


def apply_rus(train_data, test_data, val_data, random_state):
    rus = RandomUnderSampler(random_state=random_state)
    X_train = train_data.drop(columns=['TARGET'])
    y_train = train_data['TARGET']
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    # Reconstruct the training DataFrame
    train_data = pd.concat(
        [
            pd.DataFrame(X_resampled, columns=X_train.columns),
            pd.Series(y_resampled, name='TARGET')
        ],
        axis=1
    )

    log(
        logging.WARNING,
        f"Class distribution after applying RUS: {train_data['TARGET'].value_counts()}"
    )

    test_data = test_data[train_data.columns]
    val_data = val_data[train_data.columns]

    return train_data, test_data, val_data


def apply_pca(train_data, test_data, val_data, pca):
    log(
        logging.WARNING,
        f"\nPCA: {pca}"
    )
    pca_model = PCA(n_components=pca)

    X_train = train_data.drop(columns=['TARGET']).reset_index(drop=True)
    y_train = train_data['TARGET'].reset_index(drop=True)
    train_data = pd.DataFrame(pca_model.fit_transform(X_train))
    train_data['TARGET'] = y_train

    X_test = test_data.drop(columns=['TARGET']).reset_index(drop=True)
    y_test = test_data['TARGET'].reset_index(drop=True)
    test_data = pd.DataFrame(pca_model.transform(X_test))
    test_data['TARGET'] = y_test

    X_val = val_data.drop(columns=['TARGET']).reset_index(drop=True)
    y_val = val_data['TARGET'].reset_index(drop=True)
    val_data = pd.DataFrame(pca_model.transform(X_val))
    val_data['TARGET'] = y_val

    log(
        logging.WARNING,
        f"\nAfter PCA shape is: {train_data.shape[1]}"
    )

    test_data = test_data[train_data.columns]
    val_data = val_data[train_data.columns]

    return train_data, test_data, val_data


def apply_scaling(train_data, test_data, val_data):
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

    return train_data, test_data, val_data


def apply_kbest(train_data, test_data, val_data, features):
    train_data = train_data[features]
    test_data = test_data[features]
    val_data = val_data[features]

    return train_data, test_data, val_data


def transform_dataset_to_dmatrix(data) -> xgb.core.DMatrix:
    """Transform dataset to DMatrix format for xgboost."""
    y = data["def_pay"]
    x = data.drop(columns=['def_pay', 'index'])
    new_data = xgb.DMatrix(x, label=y)
    return new_data
