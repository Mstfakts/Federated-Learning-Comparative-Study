import os

os.environ["config_file"] = "mlp"

from configs.config import config
from data.dataset import load_dataloader

from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from models import model
from sklearn.metrics import make_scorer, f1_score
from sklearn.decomposition import PCA
import random

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

random_state = random.randint(1, 1000)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(random_state=random_state)),
    ('classifier', model)
])

param_grid = {
    'pca__n_components': [2, 4, 8, 10, 12, 16, 18, 20],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=make_scorer(f1_score),
    verbose=3,
    cv=3,
    n_jobs=-1
)

grid_search.fit(train_data, train_label)

best_model = grid_search.best_estimator_
print("En iyi parametreler: ", best_model)

print("Doğrulama Seti Sonuçları:")
y_val_pred = best_model.predict(val_data)
print(classification_report(val_label, y_val_pred))

print("Test Seti Sonuçları:")
y_test_pred = best_model.predict(test_data)
print(classification_report(test_label, y_test_pred))