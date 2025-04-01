import os

os.environ["config_file"] = "random_forest"

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from configs.config import get_config
from data.data_loader import partition_data_loader
import warnings
import pandas as pd

warnings.filterwarnings('ignore', message='\'n_jobs\' > 1 does not have any effect')
np.random.seed(42)

# Config ayarlarını yükle
config = get_config()
config['client'] = 1
config['data']['batch_size'] = config['data']['batch_size']
config['data']['scale'] = False
config['data']['smote'] = False
config['data']['rus'] = False
config['data']['encode'] = True
config['data']['pca'] = False
config['data']['pandas'] = True


# F1 skorunu pozitif sınıf için tanımlama
def f1_score_class_1(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=1)


f1_scorer = make_scorer(f1_score_class_1)

# Veri yükleme
train_dataloader, test_dataloader, val_dataloader, num_examples = partition_data_loader(0)

# Eğitimde kullanılacak X, y
X_sample, y_sample = train_dataloader.features, train_dataloader.labels

# RandomForest için Grid Search parametreleri
param_grid = {
    'n_estimators': [100],  # Ağaç sayısı
    'max_depth': [None, 10, 20],  # Azami derinlik (None => sınırsız)
    'min_samples_split': [2, 5],  # Dallanma için gerekli minimum örnek sayısı
    'min_samples_leaf': [1],  # Yaprak düğümünde bulunması gereken minimum örnek sayısı
    'class_weight': ['balanced'],  # Sınıf dengesizliğini ele almak için
    'n_jobs': [-1],  # Çoklu işlem
    'bootstrap': [True],
    'max_features': ['sqrt'],
    'criterion': ['gini']
}

# Model nesnesi
rf_model = RandomForestClassifier(random_state=42)

# Grid Search
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=3,
    scoring=f1_scorer,  # F1 skoru üzerinden değerlendirme
    verbose=2
)

grid_search.fit(X_sample, y_sample)

# En iyi parametreleri ve skoru görme
print("En iyi parametreler:", grid_search.best_params_)
print("En iyi skor:", grid_search.best_score_)

# En iyi modeli alma
best_model = grid_search.best_estimator_

# Test ve validation setinde tahminler
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


# ROC eğrisi çizimi
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


# RandomForestClassifier, predict_proba sağlayabildiği için doğrudan olasılık kullanabiliriz
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

metrics_df = pd.DataFrame({
    'Train': train_metrics,
    'Test': test_metrics,
    'Validation': val_metrics
})

print("\nTüm Metrikler Karşılaştırması:")
print(metrics_df)
