import os

os.environ["config_file"] = "random_forest"
import pandas as pd
from configs.config import config
from data.dataset import load_data
from sklearn.metrics import classification_report
from experiments.federated_learning.random_forest.model import model
from utils.reporting_utils import print_classification_report_from_dict
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import random
from sklearn.model_selection import train_test_split

for _ in range(7):
    train_loader, test_loader, val_loader, num_examples = load_data(
        partition_id=0,
        n_partitions=1,
        batch_size=config['data']['batch_size'],
        scale=config['data']['scale'],
        smote=config['data']['smote'],
        rus=config['data']['rus'],
        encode=config['data']['encode'],
        pca=config['data']['pca'],
        ica=config['data']['ica']
    )

    train_data = train_loader.dataset.features
    train_label = train_loader.dataset.labels
    test_data = test_loader.dataset.features
    test_label = test_loader.dataset.labels
    val_data = val_loader.dataset.features
    val_label = val_loader.dataset.labels


    model.fit(train_data, train_label)

    # print("Doğrulama Seti Sonuçları:")
    # y_val_pred = model.predict(val_data)
    # test_output = classification_report(val_label, y_val_pred, output_dict=True)
    # print(test_output)

    print("Test Seti Sonuçları:")
    y_test_pred = model.predict(test_data)
    test_output = classification_report(test_label, y_test_pred, output_dict=True)
    print(test_output)

    print_classification_report_from_dict(test_output)
