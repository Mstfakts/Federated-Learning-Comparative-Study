import os

os.environ["config_file"] = "logistic_regression"

from configs.config import config
from data.dataset import load_dataloader
from utils.reporting import print_classification_report_from_dict
from sklearn.metrics import classification_report
from models.logistic_regression import model

for _ in range(7):
    # Load data for the specified partition
    train_loader, test_loader, val_loader, num_examples = load_dataloader(
        partition_id=0,
        n_partitions=1,
        batch_size=config['data']['batch_size'],
        scale=config['data']['scale'],
        smote=config['data']['smote'],
        rus=config['data']['rus'],
        encode=config['data']['encode'],
        pca=config['data']['pca']
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