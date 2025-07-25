import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


from data.dataloader import partition_data_loader

import warnings

warnings.filterwarnings('ignore', message='\'n_jobs\' > 1 does not have any effect')
np.random.seed(42)
n_components_perc = 0.90
metric = 'accuracy'
config=""
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

# X_train, y_train = get_balanced_sample(train_dataloader.features,
#                                        train_dataloader.labels,
#                                        sample_size_per_class=17000)

X_train, y_train = train_dataloader.features, train_dataloader.labels
X_test, X_val = test_dataloader.features, val_dataloader.features
y_test, y_val = test_dataloader.labels, val_dataloader.labels

# Veri ön işleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

pca_initial = PCA(random_state=42)
pca_initial.fit(X_train_scaled)

cumsum = np.cumsum(pca_initial.explained_variance_ratio_)
n_components_95 = np.argmax(cumsum >= n_components_perc) + 1

print(f"Toplam varyansın {n_components_perc}'ini korumak için gereken bileşen sayısı: {n_components_95}")

plt.plot(range(1, len(cumsum) + 1), cumsum)
plt.xlabel('Bileşen Sayısı')
plt.ylabel('Kümülatif Varyans Oranı')
plt.axhline(y=n_components_perc, color='r', linestyle='--')
plt.show()

n_components_range = list(range(2, min(n_components_95 + 1, 200), 5))

best_params = \
    {'bootstrap': True,
     'class_weight': 'balanced',
     'criterion': 'gini',
     'max_depth': 10,
     'max_features': 'sqrt',
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'n_estimators': 100,
     'n_jobs': -1}

results = []

# PCA ve model eğitimi
for n_comp in n_components_range:
    pca = PCA(n_components=n_comp, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    lr = RandomForestClassifier(**best_params)
    if metric == 'f1':
        cv_scores = cross_val_score(lr, X_train_pca, y_train, cv=5, scoring='f1')
    else:
        cv_scores = cross_val_score(lr, X_train_pca, y_train, cv=5, scoring='accuracy')

    lr.fit(X_train_pca, y_train)
    y_pred = lr.predict(X_val_pca)

    if metric == 'f1':
        f1 = f1_score(y_val, y_pred, pos_label=1)
        results.append({
            'n_components': n_comp,
            'f1_score': f1,
            'cv_scores_mean': cv_scores.mean(),
            'cv_scores_std': cv_scores.std()
        })
        print(f"n_components={n_comp}, F1 Score={f1:.4f}")
    else:
        accuracy = accuracy_score(y_val, y_pred)
        results.append({
            'n_components': n_comp,
            'accuracy': accuracy,
            'cv_scores_mean': cv_scores.mean(),
            'cv_scores_std': cv_scores.std()
        })
        print(f"n_components={n_comp}, Accuracy={accuracy:.4f}")

if metric == 'f1':
    best_result = max(results, key=lambda x: x['f1_score'])
else:
    best_result = max(results, key=lambda x: x['accuracy'])
print(f"\nEn iyi PCA sonucu:")
print(f"Bileşen sayısı: {best_result['n_components']}")

if metric == 'f1':
    print(f"F1 Score: {best_result['f1_score']:.4f}")
else:
    print(f"Accuracy Score: {best_result['accuracy']:.4f}")

# En iyi model ile test performansı
pca = PCA(n_components=best_result['n_components'], random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
X_val_pca = pca.transform(X_val_scaled)

lr = RandomForestClassifier(**best_params)
lr.fit(X_train_pca, y_train)
y_test_pred = lr.predict(X_test_pca)

print("\nTest seti performansı:")
print(classification_report(y_test, y_test_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Test Seti Confusion Matrix')
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.show()

# En iyi parametreler ile model eğitim ve testi yap
y_val_pred = lr.predict(X_val_pca)

print("\nVal seti performansı:")
print(classification_report(y_val, y_val_pred))
