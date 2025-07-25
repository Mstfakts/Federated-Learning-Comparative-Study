import os

from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


from data.dataloader import partition_data_loader
import warnings

warnings.filterwarnings('ignore', message='\'n_jobs\' > 1 does not have any effect')
np.random.seed(42)
config=""
config['client'] = 1
config['data']['batch_size'] = config['data']['batch_size']
config['data']['scale'] = False
config['data']['smote'] = False
config['data']['rus'] = False
config['data']['encode'] = True
config['data']['pca'] = False

config['data']['pandas'] = True


def get_balanced_sample(features, labels, sample_size_per_class=1000):
    X_list = []
    y_list = []
    class_counts = {}

    # Tüm veriyi tek seferde işleyeceğiz
    for i, y in enumerate(labels):
        y = int(y)  # sınıf etiketini integer'a çevir
        if y not in class_counts:
            class_counts[y] = 0
        if class_counts[y] < sample_size_per_class:
            X_list.append(features[i:i + 1])
            y_list.append([y])
            class_counts[y] += 1

        # Tüm sınıflar için yeterli örnek toplandıysa döngüyü bitir
        if all(count >= sample_size_per_class for count in class_counts.values()):
            break

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # Karıştır
    indices = np.random.permutation(len(y))
    return X[indices], y[indices]


def f1_score_class_1(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=1)


f1_scorer = make_scorer(f1_score_class_1)

train_dataloader, test_dataloader, val_dataloader, num_examples = partition_data_loader(0)

# X_sample, y_sample = get_balanced_sample(train_dataloader.features,
#                                          train_dataloader.labels,
#                                          sample_size_per_class=10000)

X_sample, y_sample = train_dataloader.features, train_dataloader.labels

# Grid Search parametreleri
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'class_weight': ['balanced'],
    'max_iter': [1000],
    'n_jobs': [-1]
}

param_grid = {
    'C': [0.1],
    'class_weight': ['balanced'],
    'max_iter': [500],
    'solver': ["lbfgs"],
    'warm_start': [True],
    'n_jobs': [-1],
    'penalty': ['l2'],
}

# Grid Search
lr = LogisticRegression()
grid_search = GridSearchCV(
    estimator=lr,
    param_grid=param_grid,
    cv=3,
    scoring=f1_scorer,  # veya başka bir metrik
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_sample, y_sample)

# En iyi parametreleri ve skoru görme
print("En iyi parametreler:", grid_search.best_params_)
print("En iyi skor:", grid_search.best_score_)

# En iyi modeli alma
best_model = grid_search.best_estimator_

# En iyi modelle tahmin yapma
y_test_pred = best_model.predict(test_dataloader.features)
y_val_pred = best_model.predict(val_dataloader.features)

y_test = test_dataloader.labels
y_val = val_dataloader.labels
# Test performansı
print("\nTest Seti Performansı:")
print(classification_report(y_test, y_test_pred))

# Validation performansı
print("\nValidation Seti Performansı:")
print(classification_report(y_val, y_val_pred))


# Confusion Matrix'leri görselleştirme
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Tahmin')
    plt.ylabel('Gerçek')
    plt.show()


plot_confusion_matrix(y_test, y_test_pred, 'Test Seti Confusion Matrix')
plot_confusion_matrix(y_val, y_val_pred, 'Validation Seti Confusion Matrix')

# Probability tahminlerini alma ve ROC eğrisi çizme
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(y_true, y_prob, title):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC eğrisi (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


# Olasılık tahminleri
y_test_prob = best_model.predict_proba(test_dataloader.features)[:, 1]
y_val_prob = best_model.predict_proba(val_dataloader.features)[:, 1]


plot_roc_curve(y_test, y_test_prob, 'Test Seti ROC Eğrisi')
plot_roc_curve(y_val, y_val_prob, 'Validation Seti ROC Eğrisi')


# Train, Test ve Validation performanslarını karşılaştırma
def get_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred)
    }


train_metrics = get_metrics(train_dataloader.labels, best_model.predict(train_dataloader.features))
test_metrics = get_metrics(y_test, y_test_pred)
val_metrics = get_metrics(y_val, y_val_pred)

# Metrikleri DataFrame'e çevirip karşılaştırma
import pandas as pd

metrics_df = pd.DataFrame({
    'Train': train_metrics,
    'Test': test_metrics,
    'Validation': val_metrics
})

print("\nTüm Metrikler Karşılaştırması:")
print(metrics_df)
