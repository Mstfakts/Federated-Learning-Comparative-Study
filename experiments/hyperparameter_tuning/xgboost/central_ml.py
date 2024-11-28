import os

os.environ["config_file"] = "xgboost"

from configs.config import config
from data.dataset import load_xgboost_data
from utils.reporting_utils import print_classification_report_from_dict
from sklearn.metrics import classification_report
import xgboost as xgb
def get_xgboost_model():
    params = {
        "objective": config['model']['objective'],
        "learning_rate": config['model']['learning_rate'],
        "max_depth": config['model']['max_depth'],
        "eval_metric": config['model']['eval_metric'],
        "num_parallel_tree": config['model']['num_parallel_tree'],
        "tree_method": config['model']['tree_method'],
    }
    return xgb.XGBClassifier(**params)

model = get_xgboost_model()

for _ in range(7):
    # Load data for the specified partition
    train_dmatrix, test_dmatrix, valid_dmatrix, _, _, _ = load_xgboost_data(
        partition_id=0,
        n_partitions=1,
        batch_size=config['data']['batch_size'],
        scale=config['data']['scale'],
        smote=config['data']['smote'],
        encode=config['data']['encode'],
        pca=config['data']['pca'],
        ica=config['data']['ica'],
        rus=config['data']['rus']
    )

    model.fit(train_dmatrix.get_data(), train_dmatrix.get_label())

    # print("Doğrulama Seti Sonuçları:")
    # y_val_pred = model.predict(valid_dmatrix.get_data())
    # test_output = classification_report(valid_dmatrix.get_label(), y_val_pred, output_dict=True)
    # print(test_output)

    print("Test Seti Sonuçları:")
    y_test_pred = model.predict(test_dmatrix.get_data())
    test_output = classification_report(test_dmatrix.get_label(), y_test_pred, output_dict=True)
    print(test_output)

    print_classification_report_from_dict(test_output)