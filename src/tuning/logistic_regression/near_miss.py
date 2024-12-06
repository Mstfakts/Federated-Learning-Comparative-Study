import os

os.environ["config_file"] = "logistic_regression"
import pandas as pd
from configs.config import config
from data.dataset import load_dataloader
from utils.reporting import print_classification_report_from_dict
from sklearn.metrics import classification_report
from models.logistic_regression import model
from imblearn.under_sampling import NearMiss

for _ in range(10):
    # Load data for the specified partition
    train_loader, test_loader, val_loader, num_examples = load_dataloader(
        partition_id=0,
        n_partitions=1,
        batch_size=config['data']['batch_size'],
        scale=False,
        smote=False,
        encode=False,
        pca=False
    )

    train_data = train_loader.dataset.features
    train_label = train_loader.dataset.labels
    test_data = test_loader.dataset.features
    test_label = test_loader.dataset.labels
    val_data = val_loader.dataset.features
    val_label = val_loader.dataset.labels

    features_lasso = ['AGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4']
    features_rf = ['PAY_1', 'AGE', 'LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2']
    features_rfe = ['PAY_1', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT6']
    features_mi = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5']

    selected_approach = features_rfe

    train_data = train_data[selected_approach]
    test_data = test_data[selected_approach]
    val_data = val_data[selected_approach]

    print("Orijinal veri setindeki sınıf dağılımı:")
    print(pd.Series(train_label).value_counts())

    nm = NearMiss(version=3)  # Near Miss-1 kullanılıyor, version parametresi 1, 2 veya 3 olabilir.
    train_data, train_label = nm.fit_resample(train_data, train_label)

    print("\nUnder-sampled veri setindeki sınıf dağılımı:")
    print(pd.Series(train_label).value_counts())

    model.fit(train_data, train_label)

    print("Doğrulama Seti Sonuçları:")
    y_val_pred = model.predict(val_data)
    print(classification_report(val_label, y_val_pred))

    print("Test Seti Sonuçları:")
    y_test_pred = model.predict(test_data)
    test_output = classification_report(test_label, y_test_pred, output_dict=True)
    print(test_output)

    print_classification_report_from_dict(test_output)
