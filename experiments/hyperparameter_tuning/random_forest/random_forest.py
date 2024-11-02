import os

os.environ["config_file"] = "random_forest"
from configs.config import config
from data.dataset import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

train_loader, test_loader, val_loader, num_examples = load_data(
    partition_id=0,
    n_partitions=1,
    batch_size=config['data']['batch_size'],
    scale=False,
    smote=False,
    encode=False,
    pca=False,
    ica=False
)

train_data = train_loader.dataset.features
train_label = train_loader.dataset.labels
test_data = test_loader.dataset.features
test_label = test_loader.dataset.labels
val_data = val_loader.dataset.features
val_label = val_loader.dataset.labels

pipeline = Pipeline([
    ('classifier', RandomForestClassifier(random_state=42))
])


param_grid = {
    'classifier__n_estimators': [100, 200],  # 500 değeri çıkarıldı, daha az ağaç ile sonuç alınabilir.
    'classifier__criterion': ['gini', 'entropy'],  # 'log_loss' kaldırıldı, çünkü 'gini' ve 'entropy' daha yaygın.
    'classifier__max_depth': [None, 10, 20],  # Daha derin ağaçlar yerine sınırlı değerler seçildi.
    'classifier__min_samples_split': [2, 5],  # 10 değeri çıkarıldı.
    'classifier__min_samples_leaf': [1, 2, 4],  # 4 değeri çıkarıldı.
    'classifier__max_features': ['sqrt', 'log2'],  # `None` kaldırıldı, bu iki değer en çok kullanılanlar.
    'classifier__bootstrap': [True],  # Bootstrap genellikle kullanılır; False seçeneği çıkarıldı.
    'classifier__class_weight': [None, 'balanced']  # 'balanced' çıkarıldı; performans üzerinde minimal etkisi olabilir.
}

grid_rf_clf = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='accuracy',
    n_jobs=-1,
    verbose=3,
    cv=3
)

grid_rf_clf.fit(train_data, train_label)
best_model = grid_rf_clf.best_estimator_
print("En iyi parametreler: ", grid_rf_clf.best_params_)

print("Doğrulama Seti Sonuçları:")
y_val_pred = best_model.predict(val_data)
print(classification_report(val_label, y_val_pred))

print("Test Seti Sonuçları:")
y_test_pred = best_model.predict(test_data)
test_output = classification_report(test_label, y_test_pred, output_dict=True)
print(test_output)




