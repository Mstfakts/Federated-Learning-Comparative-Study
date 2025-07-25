import gc
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import resample
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import warnings
warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd
import seaborn as sns
from lofo import Dataset, LOFOImportance, plot_importance
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def plot_variance_distribution(df):
    import matplotlib.pyplot as plt
    variances = df.var().sort_values()
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(variances)), variances)
    plt.xlabel('Özellik İndeksi')
    plt.ylabel('Varyans')
    plt.title('Özelliklerin Varyans Dağılımı')
    plt.yscale('log')  # Log scale kullanarak dağılımı daha iyi görebiliriz
    plt.grid(True)
    plt.show()
    # Bazı istatistikler
    print("\nVaryans İstatistikleri:")
    print(variances.describe())


def remove_low_variance_features(df, threshold=0.01):
    # Varyans hesaplama
    variances = df.var()

    # Threshold'dan düşük varyansa sahip sütunları bulma
    low_variance_columns = variances[variances < threshold].index

    # Bu sütunları çıkarma
    df_filtered = df.drop(columns=low_variance_columns)

    print(f"Çıkarılan sütun sayısı: {len(low_variance_columns)}")
    print("Çıkarılan sütunlar:")
    print(low_variance_columns.tolist())

    return df_filtered

def basic_metrics_scorer(y_true, y_pred):
    """
    Temel sınıflandırma metriklerini hesaplayan scorer
    """
    # Temel metrikleri hesapla
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    # Sonuçları kaydet
    if not hasattr(basic_metrics_scorer, 'results'):
        basic_metrics_scorer.results = []

    basic_metrics_scorer.results.append({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

    # F1 skorunu döndür (veya başka bir metriği tercih edebilirsiniz)
    return f1

def fast_imputation(df):
    df_imputed = df.copy()

    # Sayısal değişkenleri seç
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for idx, col in enumerate(numeric_cols):

        print(f" {idx}. işlem yapılan: ", col)

        clean_data = df[[col]].replace([np.inf, -np.inf], None)
        missing_ratio = clean_data[col].isnull().mean()

        if missing_ratio == 0:
            df_imputed[col] = clean_data
            continue

        if missing_ratio < 0.05:
            # Az sayıda eksik değer için medyan
            df_imputed[col] = clean_data.fillna({col: clean_data.median().to_list()[0]})

        elif missing_ratio < 0.25:
            # Orta eksik -> SimpleImputer
            imp = SimpleImputer(strategy='median')
            df_imputed[col] = imp.fit_transform(clean_data)

        elif missing_ratio < 0.4:
            # Yüksek eksik -> Interpolasyon
            df_imputed[col] = clean_data.interpolate(method='linear')

        elif missing_ratio < 0.7:
            # Çok yüksek eksik -> MICE (az iterasyon)
            imp = IterativeImputer(max_iter=5)
            df_imputed[col] = imp.fit_transform(clean_data)

        else:
            # Aşırı yüksek eksik -> Sil
            df_imputed = df_imputed.drop(columns=col)
            numeric_cols = numeric_cols.drop(col)

    return df_imputed


def analyze_nulls(df):
    null_counts = df.isnull().sum()
    null_percentages = (df.isnull().sum() / len(df)) * 100

    null_info = pd.DataFrame({
        'Null Count': null_counts,
        'Null Percentage': null_percentages
    }).sort_values('Null Percentage', ascending=False)

    return null_info[null_info['Null Count'] > 0]


def analyze_infinity(df):
    # Sonsuz değerleri tespit et (hem pozitif hem negatif)
    inf_counts = (np.isinf(df)).sum()
    inf_percentages = (np.isinf(df).sum() / len(df)) * 100

    inf_info = pd.DataFrame({
        'Infinity Count': inf_counts,
        'Infinity Percentage': inf_percentages
    }).sort_values('Infinity Percentage', ascending=False)

    # Sadece infinity değeri içeren sütunları göster
    return inf_info[inf_info['Infinity Count'] > 0]


ROOT = "/Users/mustafaaktas/Desktop/case/home-credit-default-risk/"
RANDOM_STATE = 57
TARGET_COL = 'TARGET'

# data = pd.read_csv(ROOT + "mergetrain.csv")
#
# # Null ve Inf analizi yapalım
# print("Null Analizi:")
# print(analyze_nulls(data))
# print("Inf Analizi:")
# print(analyze_infinity(data))
#
# # HIzlı null doldurma işlemini uygulayalım
# data_imputed = fast_imputation(data)
#
# # Null doldurma sonrası kontrol
# print("\nNull Doldurma Sonrası Kontrol:")
# print(analyze_nulls(data_imputed))
# print(analyze_infinity(data_imputed))
#
# data_imputed.to_csv(ROOT + 'mergetrain_imputed.csv', index=False)
#
# del data_imputed
# gc.collect()

data_imputed = pd.read_csv(ROOT + "mergetrain_imputed.csv")

# Ölçeklendirme
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data_imputed.columns)

plot_variance_distribution(df_scaled)

# İlk eleme
# 701den 225 feature düştük ve threshold sadece 0.01
#df_filtered_1 = remove_low_variance_features(df_scaled, threshold=0.01)
selector = VarianceThreshold(threshold=0.01)
df_filtered_1 = selector.fit_transform(df_scaled)
selected_columns = data_imputed.columns[selector.get_support(indices=True)]

data_lowvarianceremoved = data_imputed[selected_columns]
data_lowvarianceremoved.to_csv(ROOT + 'train_imputed_lowvarianceremoved225.csv', index=False)

# Kalan özelliklerin varyans dağılımını tekrar çizme
plot_variance_distribution(df_filtered_1)

# Kalan özellik sayısını görme
print(f"Kalan özellik sayısı: {df_filtered_1.shape[1]}")

# %70 (train) - %30 (temp) sonra da %20 - %10
train_data, temp_data = train_test_split(df_filtered_1, test_size=0.3, random_state=RANDOM_STATE, shuffle=True)
test_data, val_data = train_test_split(temp_data, test_size=1 / 3, random_state=RANDOM_STATE, shuffle=True)
test_data = test_data[train_data.columns]  # Make sure feature order is fix.
val_data = val_data[train_data.columns]  # Make sure feature order is fix.

# Hedef değişkeni ayırma
y_train, y_test, y_val = train_data[TARGET_COL], test_data[TARGET_COL], val_data[TARGET_COL]
feature_columns = train_data.columns.drop(TARGET_COL)
X_train, X_test, X_val = train_data[feature_columns], test_data[feature_columns], val_data[feature_columns]
print("X Training shape", X_train.shape)
print("X Testing shape", X_test.shape)
print("X Validation shape", X_val.shape)
print("Y Training shape", y_train.shape)
print("Y Testing shape", y_test.shape)
print("X Validation shape", y_val.shape)


model = LogisticRegression(
    C=0.01,
    max_iter=5000,
    penalty="l1",
    class_weight=None,
    solver="saga",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# model.fit(X_train, y_train)
#
# # Tahminler
# y_train_pred = model.predict(X_train)
# y_test_pred = model.predict(X_test)
# y_val_pred = model.predict(X_val)
#
# # Model performansını değerlendirelim
# print("\nTrain Veri Seti Sonuçları:")
# print(classification_report(y_train, y_train_pred))
#
# print("\nTest Veri Seti Sonuçları:")
# print(classification_report(y_test, y_test_pred))
#
# print("\nValidation Veri Seti Sonuçları:")
# print(classification_report(y_val, y_val_pred))

# sample_df = X_train.sample(frac=0.01, random_state=RANDOM_STATE)
# sample_df['TARGET'] = y_train.loc[sample_df.index]
# print(f"Sample'ın sınıf dengesi: {np.unique(sample_df['TARGET'], return_counts=True)}")

# Dengeli veri setini birleştirin

y_train[y_train < 0] = False
y_train[y_train > 0] = True
true_samples = X_train[y_train == True].sample(n=1000, random_state=0)
false_samples = X_train[y_train == False].sample(n=len(true_samples), random_state=0)

balanced_sample = pd.concat([true_samples, false_samples])
balanced_sample['TARGET'] = pd.concat([
    pd.Series([True] * len(true_samples), index=true_samples.index),
    pd.Series([False] * len(false_samples), index=false_samples.index)
])
sample_df = balanced_sample.sample(frac=1, random_state=0).reset_index(drop=True)
print(sample_df['TARGET'].value_counts())

dataset = Dataset(df=sample_df, target=TARGET_COL, features=X_train.columns.tolist())

# scorer = make_scorer(basic_metrics_scorer)
lofo_imp = LOFOImportance(dataset, scoring="roc_auc", model=model, cv=5, n_jobs=-1)

importance_df = lofo_imp.get_importance()

#plot_importance(importance_df, figsize=(12, 20))

plot_importance(importance_df.head(10), figsize=(8, 6))
plot_importance(importance_df.tail(10), figsize=(8, 6))


selected_features = importance_df[importance_df['importance_mean'] > 0.0001]['feature'].tolist()

X_train_with_best_features = X_train[selected_features]
X_test_with_best_features = X_test[selected_features]
X_val_with_best_features = X_val[selected_features]