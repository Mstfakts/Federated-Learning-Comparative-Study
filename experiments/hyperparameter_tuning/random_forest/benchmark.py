import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
path_ = 'C:\\Users\\B3LAB\\PycharmProjects\\FL-Benchmark\\data/UCI_Credit_Card.csv'
numerical_columns = ['LIMIT_BAL', 'AGE',
                     'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                     'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', ]
categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE',
                       'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']


data = pd.read_csv(path_)
data = data.drop('ID',axis=1)
data = data.rename(columns={'default.payment.next.month': 'default'})


# fil = (data.EDUCATION == 5) | (data.EDUCATION == 6) | (data.EDUCATION == 0)
# data.loc[fil, 'EDUCATION'] = 4
# data.loc[data.MARRIAGE == 0, 'MARRIAGE'] = 3

data = data.rename(columns={'PAY_0': 'PAY_1'})

fil = (data.PAY_1 == -2) | (data.PAY_1 == -1) | (data.PAY_1 == 0)
data.loc[fil, 'PAY_1'] = 0
fil = (data.PAY_2 == -2) | (data.PAY_2 == -1) | (data.PAY_2 == 0)
data.loc[fil, 'PAY_2'] = 0
fil = (data.PAY_3 == -2) | (data.PAY_3 == -1) | (data.PAY_3 == 0)
data.loc[fil, 'PAY_3'] = 0
fil = (data.PAY_4 == -2) | (data.PAY_4 == -1) | (data.PAY_4 == 0)
data.loc[fil, 'PAY_4'] = 0
fil = (data.PAY_5 == -2) | (data.PAY_5 == -1) | (data.PAY_5 == 0)
data.loc[fil, 'PAY_5'] = 0
fil = (data.PAY_6 == -2) | (data.PAY_6 == -1) | (data.PAY_6 == 0)
data.loc[fil, 'PAY_6'] = 0

# fil = (data.PAY_1 == 0) & (data.PAY_2 == 0) & (data.PAY_3 == 0) & (data.PAY_4 == 0) & (data.PAY_5 == 0) & (data.PAY_6 == 0) & (data.default ==1)
# data.loc[fil,'default'] = 0
#
# fil = (data.PAY_1 > 0) & (data.PAY_2 > 0) & (data.PAY_3 > 0) & (data.PAY_4 > 0) & (data.PAY_5 > 0) & (data.PAY_6 > 0) & (data.default ==0)
# data.loc[fil,'default'] = 1

# data.loc[data.PAY_1 <= 0, 'PAY_1'] = 0
# data.loc[data.PAY_2 <= 0, 'PAY_2'] = 0
# data.loc[data.PAY_3 <= 0, 'PAY_3'] = 0
# data.loc[data.PAY_4 <= 0, 'PAY_4'] = 0
# data.loc[data.PAY_5 <= 0, 'PAY_5'] = 0
# data.loc[data.PAY_6 <= 0, 'PAY_6'] = 0
#
# data.loc[data.PAY_1 > 0, 'PAY_1'] = 1
# data.loc[data.PAY_2 > 0, 'PAY_2'] = 1
# data.loc[data.PAY_3 > 0, 'PAY_3'] = 1
# data.loc[data.PAY_4 > 0, 'PAY_4'] = 1
# data.loc[data.PAY_5 > 0, 'PAY_5'] = 1
# data.loc[data.PAY_6 > 0, 'PAY_6'] = 1


# fil = (data.BILL_AMT1 < 0)
# data.loc[fil,'BILL_AMT1'] = 0
#
# fil = (data.BILL_AMT2 < 0)
# data.loc[fil,'BILL_AMT2'] = 0
#
# fil = (data.BILL_AMT3 < 0)
# data.loc[fil,'BILL_AMT3'] = 0
#
# fil = (data.BILL_AMT4 < 0)
# data.loc[fil,'BILL_AMT4'] = 0
#
# fil = (data.BILL_AMT5 < 0)
# data.loc[fil,'BILL_AMT5'] = 0
#
# fil = (data.BILL_AMT6 < 0)
# data.loc[fil,'BILL_AMT6'] = 0

# scaler = MinMaxScaler()
# data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

target = 'default'
X = data.drop(target, axis=1)
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# smt = SMOTE(random_state=33)
# X_train, y_train = smt.fit_resample(X_train, y_train.ravel())

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk Skoru: {accuracy:.2f}")

# Sınıflandırma raporu oluştur
print(classification_report(y_test, y_pred))
