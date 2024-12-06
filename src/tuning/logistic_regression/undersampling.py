import os

os.environ["config_file"] = "logistic_regression"
import random
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from configs.config import config
from data.dataset import load_dataloader
from utils.reporting import print_classification_report_from_dict
from sklearn.metrics import classification_report
from models.logistic_regression import model

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

    # Yeni dengeli veri setini kontrol etme
    print("Orijinal veri setindeki sınıf dağılımı:")
    print(pd.Series(train_label).value_counts())

    random_state = random.randint(1, 1000)
    rus = RandomUnderSampler(random_state=random_state)
    train_data, train_label = rus.fit_resample(train_data, train_label)

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
