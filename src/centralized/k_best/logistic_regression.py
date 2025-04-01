import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2

os.environ["config_file"] = "logistic_regression"
from configs.config import get_config
from data.data_loader import partition_data_loader

import warnings

warnings.filterwarnings('ignore', message='\'n_jobs\' > 1 does not have any effect')

np.random.seed(42)
metric = 'accuracy'
config = get_config()
config['client'] = 1
config['data']['batch_size'] = config['data']['batch_size']
config['data']['scale'] = False
config['data']['smote'] = False
config['data']['rus'] = False
config['data']['encode'] = True
config['data']['pca'] = False
config['data']['pandas'] = True


def get_balanced_sample(features, labels, sample_size_per_class=None):
    unique_classes = np.unique(labels)
    if sample_size_per_class is None:
        sample_size_per_class = min([sum(labels == c) for c in unique_classes])

    total_samples = sample_size_per_class * len(unique_classes)
    X = np.zeros((total_samples, features.shape[1]))
    y = np.zeros(total_samples)

    current_idx = 0
    for c in unique_classes:
        class_indices = np.where(labels == c)[0]
        selected_indices = np.random.choice(class_indices, sample_size_per_class, replace=False)

        end_idx = current_idx + sample_size_per_class
        X[current_idx:end_idx] = features[selected_indices]
        y[current_idx:end_idx] = c
        current_idx = end_idx

    indices = np.random.permutation(len(y))
    return X[indices], y[indices]


train_dataloader, test_dataloader, val_dataloader, num_examples = partition_data_loader(0)

X_train, y_train = train_dataloader.features, train_dataloader.labels
X_test, X_val = test_dataloader.features, val_dataloader.features
y_test, y_val = test_dataloader.labels, val_dataloader.labels

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

# K aralığını belirleme
k_range = [5, 10, 15, 20, 25, 30]

best_params = {
    'C': 0.1,
    'class_weight': 'balanced',
    'max_iter': 500,
    'n_jobs': -1,
    'penalty': 'l2',
    'solver': 'lbfgs',
    'warm_start': True,
    'random_state': 42
}

results = []

for k in k_range:
    selector = SelectKBest(score_func=chi2, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_val_selected = selector.transform(X_val_scaled)

    lr = LogisticRegression(**best_params)
    if metric == 'f1':
        cv_scores = cross_val_score(lr, X_train_selected, y_train, cv=5, scoring='f1')
    else:
        cv_scores = cross_val_score(lr, X_train_selected, y_train, cv=5, scoring='accuracy')

    lr.fit(X_train_selected, y_train)
    y_pred = lr.predict(X_val_selected)

    if metric == 'f1':
        f1 = f1_score(y_val, y_pred, pos_label=1)
        results.append({
            'k_features': k,
            'f1_score': f1,
            'cv_scores_mean': cv_scores.mean(),
            'cv_scores_std': cv_scores.std()
        })
        print(f"k_features={k}, F1 Score={f1:.4f}")
    else:
        accuracy = accuracy_score(y_val, y_pred)
        results.append({
            'k_features': k,
            'accuracy': accuracy,
            'cv_scores_mean': cv_scores.mean(),
            'cv_scores_std': cv_scores.std()
        })
        print(f"k_features={k}, Accuracy={accuracy:.4f}")

if metric == 'f1':
    best_result = max(results, key=lambda x: x['f1_score'])
else:
    best_result = max(results, key=lambda x: x['accuracy'])

print(f"\nEn iyi feature selection sonucu:")
print(f"Özellik sayısı: {best_result['k_features']}")
if metric == 'f1':
    print(f"F1 Score: {best_result['f1_score']:.4f}")
else:
    print(f"Accuracy Score: {best_result['accuracy']:.4f}")

# En iyi k değeri ile test performansı
selector = SelectKBest(score_func=f_classif, k=best_result['k_features'])
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)
X_val_selected = selector.transform(X_val_scaled)

lr = LogisticRegression(**best_params)
lr.fit(X_train_selected, y_train)
y_test_pred = lr.predict(X_test_selected)

print("\nTest seti performansı:")
print(classification_report(y_test, y_test_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Test Seti Confusion Matrix')
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.show()

# En iyi parametreler ile validation seti performansı
y_val_pred = lr.predict(X_val_selected)

print("\nVal seti performansı:")
print(classification_report(y_val, y_val_pred))

support_mask = selector.get_support()
selected_indices = np.where(support_mask)[0]

print("Seçilen sütunların indeksleri:", selected_indices)

feature_names = train_dataloader.data.columns.drop("TARGET")
selected_feature_names = [feature_names[i] for i in selected_indices]

print("Seçilen sütunların isimleri:", selected_feature_names)

import pandas as pd
# Feature scores ile feature isimlerini birleştirme
feature_scores = pd.DataFrame({
    'Feature': feature_names,
    'Score': selector.scores_
})
feature_scores = feature_scores.sort_values('Score', ascending=False)

# Görselleştirme
plt.figure(figsize=(15, 8))
sns.barplot(data=feature_scores.head(best_result['k_features']), x='Score', y='Feature')
plt.title(f"Top {best_result['k_features']} En Önemli Features")
plt.xlabel('F-score')
plt.tight_layout()
plt.show()
# ['DAYS_BIRTH', 'DAYS_ID_PUBLISH', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_LAST_PHONE_CHANGE', 'CODE_GENDER_F', 'CODE_GENDER_M', 'NAME_INCOME_TYPE_Working', 'NAME_EDUCATION_TYPE_Higher education', 'NAME_EDUCATION_TYPE_Secondary / secondary special', 'preapp_CODE_REJECT_REASON_HC_mean', 'preapp_NAME_PRODUCT_TYPE_walk-in_mean', 'preapp_NAME_CONTRACT_STATUS_Refused_mean', 'preapp_NFLAG_INSURED_ON_APPROVAL_Missing_mean', 'preapp_NAME_CONTRACT_STATUS_Approved_mean', 'preapp_CODE_REJECT_REASON_XAP_mean', 'preapp_DAYS_DECISION_min', 'preapp_DAYS_FIRST_DRAWING_count', 'preapp_DAYS_FIRST_DUE_min', 'preapp_DAYS_LAST_DUE_1ST_VERSION_min', 'preapp_DAYS_LAST_DUE_min', 'client_installments_DAYS_ENTRY_PAYMENT_min_min', 'client_installments_DAYS_INSTALMENT_min_min', 'client_installments_DAYS_INSTALMENT_mean_min', 'client_installments_DAYS_ENTRY_PAYMENT_mean_min', 'client_installments_DAYS_ENTRY_PAYMENT_max_min', 'client_installments_DAYS_INSTALMENT_max_min']