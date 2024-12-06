import os

os.environ["config_file"] = "random_forest"

from configs.config import config
from data.dataset import load_dataloader

# from experiments.federated_learning.logistic_regression.model import model

# Load data for the specified partition
train_loader, test_loader, val_loader, num_examples = load_dataloader(
    partition_id=0,
    n_partitions=1,
    batch_size=config['data']['batch_size'],
    scale=True,
    smote=False,
    encode=True,
    pca=False
)

X = train_loader.dataset.data
y = X['def_pay']
X = X.drop(columns=['index', 'def_pay'])

#
# X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])  # Özellik adlarını otomatik oluşturuyoruz
# y_df = pd.DataFrame(y, columns=['label'])  # Hedef değişken için sütun adı
#
#
# random_state = random.randint(1, 1000)
#
# """
# L1 Regularizaiton (LASSO)
# """
# from sklearn.linear_model import Lasso
# from sklearn.feature_selection import SelectFromModel
# # Lasso kullanarak özellik seçimi
# lasso = Lasso(alpha=0.01)
# lasso.fit(X, y)
#
# model = SelectFromModel(lasso, prefit=True)
# selected_features = X.columns[model.get_support()]
#
# print("LASSO - Seçilen Özellikler:")
# print(selected_features)
#
# """
# Tree-based Models
# """
# from sklearn.ensemble import RandomForestClassifier
# # Random Forest ile model oluşturma
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X, y)
#
# # Özellik önemlerini elde etme
# importances = rf.feature_importances_
#
# # Seçilen özellikleri sıralama
# feature_importances = pd.DataFrame({'Özellik': X.columns, 'Önem': importances})
# feature_importances = feature_importances.sort_values(by='Önem', ascending=False)
#
# print("RandomForestClassifier - Özelliklerin Önem Dereceleri:")
# print(feature_importances)
#
# """
# Recursive Feature Elimination (RFE)
# """
# from sklearn.feature_selection import RFE
# from sklearn.ensemble import RandomForestClassifier
#
# # Random Forest kullanarak RFE uygulama
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# rfe = RFE(estimator=model, n_features_to_select=5)
# rfe.fit(X, y)
#
# # Seçilen özelliklerin maskesi
# mask = rfe.support_
#
# # Seçilen özelliklerin isimleri
# selected_features = X.columns[mask]
#
# print("RFE - Seçilen Özellikler:")
# print(selected_features)
#
# """
# Mutual Information
# """
# from sklearn.feature_selection import mutual_info_classif
#
# # Karşılıklı bilgi kullanarak özelliklerin önemini değerlendirme
# mi = mutual_info_classif(X, y, random_state=42)
#
# # Karşılıklı bilgi skorlarını elde etme
# mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
#
# print("Karşılıklı Bilgi Skorları:")
# print(mi_scores)
#

"""
Chi Squared
"""
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif

# X_new = copy.deepcopy(X)
# X_new['AGE'] = pd.cut(X_new['AGE'], bins=5, labels=[0, 1, 2, 3, 4])
# X_new['LIMIT_BAL'] = pd.cut(X['LIMIT_BAL'], bins=5, labels=[0, 1, 2, 3, 4])
#
# X_new['PAY_AMT1'] = pd.cut(X['PAY_AMT1'], bins=5, labels=[0, 1, 2, 3, 4])
# X_new['PAY_AMT2'] = pd.cut(X['PAY_AMT2'], bins=5, labels=[0, 1, 2, 3, 4])
# X_new['PAY_AMT3'] = pd.cut(X['PAY_AMT3'], bins=5, labels=[0, 1, 2, 3, 4])
# X_new['PAY_AMT4'] = pd.cut(X['PAY_AMT4'], bins=5, labels=[0, 1, 2, 3, 4])
# X_new['PAY_AMT5'] = pd.cut(X['PAY_AMT5'], bins=5, labels=[0, 1, 2, 3, 4])
# X_new['PAY_AMT6'] = pd.cut(X['PAY_AMT6'], bins=5, labels=[0, 1, 2, 3, 4])
#
# X_new['BILL_AMT1'] = pd.cut(X['BILL_AMT1'], bins=5, labels=[0, 1, 2, 3, 4])
# X_new['BILL_AMT2'] = pd.cut(X['BILL_AMT2'], bins=5, labels=[0, 1, 2, 3, 4])
# X_new['BILL_AMT3'] = pd.cut(X['BILL_AMT3'], bins=5, labels=[0, 1, 2, 3, 4])
# X_new['BILL_AMT4'] = pd.cut(X['BILL_AMT4'], bins=5, labels=[0, 1, 2, 3, 4])
# X_new['BILL_AMT5'] = pd.cut(X['BILL_AMT5'], bins=5, labels=[0, 1, 2, 3, 4])
# X_new['BILL_AMT6'] = pd.cut(X['BILL_AMT6'], bins=5, labels=[0, 1, 2, 3, 4])

# X_new['PAY_1'] += 2
# X_new['PAY_2'] += 2
# X_new['PAY_3'] += 2
# X_new['PAY_4'] += 2
# X_new['PAY_5'] += 2
# X_new['PAY_6'] += 2

categorical_features = X.columns[X.columns.str.contains('SEX|EDUCATION|MARRIAGE')].tolist()
continuous_features = [col for col in X.columns if col not in categorical_features]

selector_continuous = SelectKBest(f_classif, k=5)  # Örneğin en iyi 5 sürekli özellik
X_continuous_selected = selector_continuous.fit_transform(X[continuous_features], y)

selector_categorical = SelectKBest(chi2, k=5)  # Örneğin en iyi 5 kategorik özellik
X_categorical_selected = selector_categorical.fit_transform(X[categorical_features], y)

f_classif_scores = selector_continuous.scores_

# Skorları ve sürekli özellik isimlerini bir DataFrame'de birleştirme
continuous_scores_df = pd.DataFrame({
    'Feature': continuous_features,
    'Score': f_classif_scores
}).sort_values(by='Score', ascending=False)

# print("Sürekli özelliklerin etkisi:")
# print(continuous_scores_df)


chi2_scores = selector_categorical.scores_

# Skorları ve kategorik özellik isimlerini bir DataFrame'de birleştirme
categorical_scores_df = pd.DataFrame({
    'Feature': categorical_features,
    'Score': chi2_scores
}).sort_values(by='Score', ascending=False)

# print("Kategorik özelliklerin etkisi:")
# print(categorical_scores_df)

all_features_df = pd.concat([continuous_scores_df, categorical_scores_df])

# Skorları büyükten küçüğe sıralama
all_features_df_sorted = all_features_df.sort_values(by='Score', ascending=False)

print("Tüm özelliklerin hedef değişken üzerindeki etkisi:")
print(all_features_df_sorted)








