# Gerçek verileri elle girmek için "real_data" sözlüğünü tanımlayın.
# Sözlüğün yapısı:
# { "ML_Combo": {
#       "Precision": { "Centralized": {"TAIWAN": value, "HCDR": value, "HMEQ": value},
#                      "FedAvg": {...},
#                      "FedF1": {...} },
#       "Recall": { ... },
#       "Accuracy": { ... },
#       "F1": { ... }
#    },
#   "XGB": { ... },
#   "MLP": { ... },
#   ...
# }

# "LR": {
#         "Precision": {
#             "Centralized": {"TAIWAN": 0.0, "HCDR": 0.0, "HMEQ": 0.0},
#             "FedAvg": {"TAIWAN": 0.0, "HCDR": 0.0, "HMEQ": 0.0},
#             "FedF1": {"TAIWAN": 0.0, "HCDR": 0.0, "HMEQ": 0.0}
#         },
#         "Recall": {
#             "Centralized": {"TAIWAN": 0.0, "HCDR": 0.0, "HMEQ": 0.0},
#             "FedAvg": {"TAIWAN": 0.0, "HCDR": 0.0, "HMEQ": 0.0},
#             "FedF1": {"TAIWAN": 0.0, "HCDR": 0.0, "HMEQ": 0.0}
#         },
#         "Accuracy": {
#             "Centralized": {"TAIWAN": 0.0, "HCDR": 0.0, "HMEQ": 0.0},
#             "FedAvg": {"TAIWAN": 0.0, "HCDR": 0.0, "HMEQ": 0.0},
#             "FedF1": {"TAIWAN": 0.0, "HCDR": 0.0, "HMEQ": 0.0}
#         },
#         "F1": {
#             "Centralized": {"TAIWAN": 0.0, "HCDR": 0.0, "HMEQ": 0.0},
#             "FedAvg": {"TAIWAN": 0.0, "HCDR": 0.0, "HMEQ": 0.0},
#             "FedF1": {"TAIWAN": 0.0, "HCDR": 0.0, "HMEQ": 0.0}
#         }
#     },


# Aşağıdaki örnekte, yalnızca "LR" için örnek değerler verdim. Diğer modelleri benzer şekilde ekleyin.

import pandas as pd


def create_model_key(ml_model, sampling, feature_reduction):
    """
    Sözlüğün anahtar ismini "LR", "LR-PCA", "LR-RUS-KBEST" vb. şekilde oluşturur.
    """
    parts = [ml_model.upper()]
    if sampling.lower() not in ["none", "n", ""]:
        parts.append(sampling.upper())
    if feature_reduction.lower() not in ["none", "n", ""]:
        parts.append(feature_reduction.upper())
    return "-".join(parts)


def fill_real_data_from_excel(df):
    """
    df, Excel'den okunmuş pandas DataFrame (örn. df = pd.read_excel("..."))
    Sütunlar:
      'Dataset', 'ML Model', 'Data Sampling', 'Aggregation',
      'Feature Reduction', 'Precision', 'Recall', 'Accuracy', 'F1'
    """
    df = df.fillna(0)
    real_data = {}

    for idx, row in df.iterrows():
        dataset = row["Dataset"]  # Örn. "TAIWAN"
        ml_model = row["ML Model"]  # Örn. "LR"
        data_sampling = row["Data Sampling"]  # Örn. "SMOTE"
        aggregation = row["Aggregation"]  # Örn. "Centralized"
        feature_reduction = row["Feature Reduction"]  # Örn. "PCA"

        precision = row["Precision"]
        recall = row["Recall"]
        accuracy = row["Accuracy"]
        f1 = row["F1"]

        # 1) Modelin anahtar ismi (örn. "LR-SMOTE-PCA"):
        model_key = create_model_key(ml_model, data_sampling, feature_reduction)

        # 2) Sözlükte yoksa katmanları oluştur:
        if model_key not in real_data:
            real_data[model_key] = {
                "Precision": {},
                "Recall": {},
                "Accuracy": {},
                "F1": {}
            }
        # Örn. real_data["LR-SMOTE-PCA"]["Precision"] => Aggregation => Dataset => Value

        for metric_name, metric_val in zip(["Precision", "Recall", "Accuracy", "F1"],
                                           [precision, recall, accuracy, f1]):

            if metric_name not in real_data[model_key]:
                real_data[model_key][metric_name] = {}
            if aggregation not in real_data[model_key][metric_name]:
                real_data[model_key][metric_name][aggregation] = {}

            # 3) Değeri ata
            # Örn. real_data["LR-SMOTE-PCA"]["Precision"]["Centralized"]["TAIWAN"] = 0.72
            real_data[model_key][metric_name][aggregation][dataset] = metric_val

    return real_data


# Örnek kullanım:
df = pd.read_excel("/Users/mustafaaktas/Desktop/TumDeneySonuclari.xlsx")
real_data = fill_real_data_from_excel(df)
