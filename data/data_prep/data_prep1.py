import gc

import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score,StratifiedKFold,RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score,mean_squared_error, r2_score,classification_report
from sklearn.preprocessing import LabelBinarizer, LabelEncoder,StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from ycimpute.imputer import EM
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import re
import gc

ROOT = "/Users/mustafaaktas/Desktop/case/home-credit-default-risk/"


def create_denormalized_tables(df_main, df_merge, on='SK_ID_CURR'):
    df_main = df_main.merge(right=df_merge.reset_index(), how='left', on=on)
    return df_main


def load_all_data():
    app_train_ = pd.read_csv(ROOT + "application_train.csv")
    # app_test_ = pd.read_csv(ROOT + "application_test.csv")  # Omit Test file, it does not contain label.
    subm_ = pd.read_csv(ROOT + "sample_submission.csv")
    pos_cash_ = pd.read_csv(ROOT + 'POS_CASH_balance.csv')
    credit_card_ = pd.read_csv(ROOT + 'credit_card_balance.csv')
    bureau_ = pd.read_csv(ROOT + 'bureau.csv')
    bureau_balance_ = pd.read_csv(ROOT + 'bureau_balance.csv')
    previous_app_ = pd.read_csv(ROOT + 'previous_application.csv')
    install_payments_ = pd.read_csv(ROOT + 'installments_payments.csv')

    return app_train_, subm_, pos_cash_, credit_card_, bureau_, bureau_balance_, previous_app_, install_payments_


def missing_data(data):
    analysis_results = {
        'Column Name': data.columns,
        'Total': data.isnull().sum(),
        'Percent': (data.isnull().sum() / len(data)) * 100,
        'dtypes': data.dtypes
    }
    analysis_df = pd.DataFrame(analysis_results)
    analysis_df = analysis_df.sort_values('Percent', ascending=False)
    analysis_df['Percent'] = analysis_df['Percent'].round(2)
    analysis_df = analysis_df.set_index('Column Name')
    return analysis_df


def summary(data):
    stats_df = data.describe().T.round(2)
    stats_df['count'] = stats_df['count'].astype(int)
    stats_df.columns = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    return stats_df


def correlation(data):
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

    target_correlations = data[numeric_columns].corrwith(data['TARGET']).sort_values(ascending=False)

    target_corr_df = pd.DataFrame({
        'Feature': target_correlations.index,
        'Correlation': target_correlations.values
    })
    target_corr_df['Correlation'] = target_corr_df['Correlation'].round(3)
    target_corr_df = target_corr_df.sort_values('Correlation', ascending=False)

    print("Top correlations with TARGET:")
    print(target_corr_df.head(10))
    print("\nBottom correlations with TARGET:")
    print(target_corr_df.tail(10))


def convert_types(df, print_info=False):
    original_memory = df.memory_usage().sum()

    # Iterate through each column
    for c in df:

        # Convert ids and booleans to integers
        if 'SK_ID' in c:
            df[c] = df[c].fillna(0).astype(np.int32)

        # Convert objects to category
        elif (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype('category')

        # Booleans mapped to integers
        elif list(df[c].unique()) == [1, 0]:
            df[c] = df[c].astype(bool)

        # Float64 to float32
        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)

        # Int64 to int32
        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)

    new_memory = df.memory_usage().sum()

    if print_info:
        print(f'Original Memory Usage: {round(original_memory / 1e9, 2)} gb.')
        print(f'New Memory Usage: {round(new_memory / 1e9, 2)} gb.')

    return df


def aggregate_numerical_features(df, group_var, df_name):
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns=col)

    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids

    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()

    columns = [group_var]

    for var in agg.columns.levels[0]:
        if var != group_var:
            for stat in agg.columns.levels[1][:-1]:
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns
    return agg


def agg_numeric(df, parent_var, df_name):
    """
    Groups and aggregates the numeric values in a child dataframe
    by the parent variable.

    Parameters
    --------
        df (dataframe):
            the child dataframe to calculate the statistics on
        parent_var (string):
            the parent variable used for grouping and aggregating
        df_name (string):
            the variable used to rename the columns

    Return
    --------
        agg (dataframe):
            a dataframe with the statistics aggregated by the `parent_var` for
            all numeric columns. Each observation of the parent variable will have
            one row in the dataframe with the parent variable as the index.
            The columns are also renamed using the `df_name`. Columns with all duplicate
            values are removed.

    """

    # Remove id variables other than grouping variable
    for col in df:
        if col != parent_var and 'SK_ID' in col:
            df = df.drop(columns=col)

    # Only want the numeric variables
    parent_ids = df[parent_var].copy()
    numeric_df = df.select_dtypes('number').copy()
    numeric_df[parent_var] = parent_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(parent_var).agg(['count', 'mean', 'max', 'min', 'sum'])

    # Need to create new column names
    columns = []

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        if var != parent_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns

    # Remove the columns with all redundant values
    _, idx = np.unique(agg, axis=1, return_index=True)
    agg = agg.iloc[:, idx]

    return agg


def agg_categorical(df, parent_var, df_name):
    """
    Aggregates the categorical features in a child dataframe
    for each observation of the parent variable.

    Parameters
    --------
    df : dataframe
        The dataframe to calculate the value counts for.

    parent_var : string
        The variable by which to group and aggregate the dataframe. For each unique
        value of this variable, the final dataframe will have one row

    df_name : string
        Variable added to the front of column names to keep track of columns


    Return
    --------
    categorical : dataframe
        A dataframe with aggregated statistics for each observation of the parent_var
        The columns are also renamed and columns with duplicate values are removed.

    """

    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('category'))

    # Make sure to put the identifying id on the column
    categorical[parent_var] = df[parent_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(parent_var).agg(['sum', 'count', 'mean'])

    column_names = []

    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['sum', 'count', 'mean']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))

    categorical.columns = column_names

    # Remove duplicate columns by values
    _, idx = np.unique(categorical, axis=1, return_index=True)
    categorical = categorical.iloc[:, idx]

    return categorical


def aggregate_client(df, group_vars, df_names):
    """Aggregate a dataframe with data at the loan level
    at the client level

    Args:
        df (dataframe): data at the loan level
        group_vars (list of two strings): grouping variables for the loan
        and then the client (example ['SK_ID_PREV', 'SK_ID_CURR'])
        names (list of two strings): names to call the resulting columns
        (example ['cash', 'client'])

    Returns:
        df_client (dataframe): aggregated numeric stats at the client level.
        Each client will have a single row with all the numeric data aggregated
    """

    # Aggregate the numeric columns
    df_agg = agg_numeric(df, parent_var=group_vars[0], df_name=df_names[0])

    # If there are categorical variables
    if any(df.dtypes == 'category'):

        # Count the categorical columns
        df_counts = agg_categorical(df, parent_var=group_vars[0], df_name=df_names[0])

        # Merge the numeric and categorical
        df_by_loan = df_counts.merge(df_agg, on=group_vars[0], how='outer')

        gc.enable()
        del df_agg, df_counts
        gc.collect()

        # Merge to get the client id in dataframe
        df_by_loan = df_by_loan.merge(df[[group_vars[0], group_vars[1]]], on=group_vars[0], how='left')

        # Remove the loan id
        df_by_loan = df_by_loan.drop(columns=[group_vars[0]])

        # Aggregate numeric stats by column
        df_by_client = agg_numeric(df_by_loan, parent_var=group_vars[1], df_name=df_names[1])


    # No categorical variables
    else:
        # Merge to get the client id in dataframe
        df_by_loan = df_agg.merge(df[[group_vars[0], group_vars[1]]], on=group_vars[0], how='left')

        gc.enable()
        del df_agg
        gc.collect()

        # Remove the loan id
        df_by_loan = df_by_loan.drop(columns=[group_vars[0]])

        # Aggregate numeric stats by column
        df_by_client = agg_numeric(df_by_loan, parent_var=group_vars[1], df_name=df_names[1])

    # Memory management
    gc.enable()
    del df, df_by_loan
    gc.collect()

    return df_by_client


def count_categorical(df, group_var, df_name):
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))

    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])

    column_names = []

    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))

    categorical.columns = column_names

    return categorical


app_train, subm, pos_cash, credit_card, bureau, bureau_balance, previous_app, install_payments = load_all_data()

""" EDA - app_train """
# missing_data(app_train)
# summary(app_train)
# correlation(app_train)
#
# """ Feature Engineering """
# # Contact details
# app_train['CONTACT_DETAILS'] = app_train[['FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']].sum(axis=1)
# app_train.drop(app_train[['FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']], axis=1, inplace=True)
#
# # Wrong address
# app_train['WRONG_ADDRESS'] = app_train[['REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
#                                         'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
#                                         'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']].sum(axis=1)
# app_train.drop(app_train[['REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
#                           'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
#                           'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']], axis=1, inplace=True)
#
# # Documentation
# app_train['DOCUMENT_SUM_NEGATIVE_CURR'] = app_train[['FLAG_DOCUMENT_' + str(i) for i in range(4, 20)]].sum(axis=1) / 16
# app_train['DOCUMENT_SUM_POSITIVE_CORR'] = (app_train[['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',
#                                                       'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']].sum(axis=1)) / 4
# app_train.drop(app_train[['FLAG_DOCUMENT_' + str(i) for i in range(2, 22)]], axis=1, inplace=True)
#
# # Building
# app_train['BUILDING_FEATURES'] = ((app_train.loc[:, 'APARTMENTS_AVG':'NONLIVINGAREA_AVG'].sum(axis=1)) *
#                                   (3 - app_train['REGION_RATING_CLIENT_W_CITY']))
# app_train.drop(app_train.loc[:, 'APARTMENTS_AVG':'NONLIVINGAREA_MEDI'], axis=1, inplace=True)
#
# # Days
# app_train["DAYS_BIRTH"] = app_train["DAYS_BIRTH"] * -1 / 365
# app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
# app_train['DAYS_EMPLOYED'] = app_train['DAYS_EMPLOYED'] * -1 / 365
# app_train['DAYS_REGISTRATION'] = app_train['DAYS_REGISTRATION'] * -1 / 365
# app_train['DAYS_ID_PUBLISH'] = app_train['DAYS_ID_PUBLISH'] * -1 / 365
#
# """ Label Encoding """
# # Categorical columns
# categorical_cols = app_train.select_dtypes(include=['object']).columns.to_list()
# df_cat = app_train[categorical_cols].copy()
# for col in list(categorical_cols):
#     df_cat[col] = df_cat[col].ffill()
#
# le = LabelEncoder()
# for col in categorical_cols:
#     if len(list(df_cat[col].unique())) <= 2:
#         le.fit(df_cat[col])
#         df_cat.loc[:, col] = le.transform(df_cat[col])
# df_cat = pd.get_dummies(df_cat)
#
# # Numeric columns
# numeric_cols = app_train.select_dtypes(include=['int64', 'float64']).columns.to_list()
# df_num = app_train[numeric_cols].copy()
# df_num.drop(['SK_ID_CURR', 'TARGET'], axis=1, inplace=True)
#
# """ Null values """
# # Nan values
# var_names = list(df_num)
# np_df_num = np.array(df_num)
# dff = EM().complete(np_df_num)
# dff = pd.DataFrame(dff, columns=var_names)
#
# """ Outliers """
# # Outliers
# clf = LocalOutlierFactor(n_neighbors=10, contamination=0.1)
#
# clf.fit_predict(dff)
# dff_scores = clf.negative_outlier_factor_
#
# threshold_value = np.sort(dff_scores)[7]  # TODO Farklı değerler dene. Bir de farklı yöntemler dene (elbow, )
# outlier_mask = dff_scores < threshold_value
#
# suppression_value = dff[dff_scores == threshold_value]
#
# outlier_data = dff[outlier_mask]
# outlier_data.to_records(index=False)
# res = outlier_data.to_records(index=False)
# res[:] = suppression_value.to_records(index=False)
# dff[outlier_mask] = pd.DataFrame(res, index=dff[outlier_mask].index)
#
# """ Final & Save """
# # Merging
# dff.insert(0, 'SK_ID_CURR', app_train['SK_ID_CURR'].values)
# dff.insert(1, 'TARGET', app_train['TARGET'].values)
# df_cat.insert(0, 'SK_ID_CURR', app_train['SK_ID_CURR'].values)
#
# app_train_preprocessed = pd.merge(dff, df_cat, on='SK_ID_CURR')
# print("Any Null Value app_train?: ", app_train_preprocessed.isnull().sum().any())
# app_train_preprocessed.to_csv(ROOT + 'app_train_preprocessed.csv')

""" EDA - install_payments """
# summary(install_payments)
# missing_data(install_payments)
#
# dfinstagg = aggregate_numerical_features(install_payments, group_var='SK_ID_CURR', df_name='instpay')
# installments_by_client = aggregate_client(
#     install_payments, group_vars=['SK_ID_PREV', 'SK_ID_CURR'], df_names=['installments', 'client'])
#
# # reduce memory usage
# dfinstagg = convert_types(installments_by_client, print_info=True)
# dfinstagg.reset_index(inplace=True)
#
# """ Final & Save """
# print("Any Null Value install_payments?: ", dfinstagg.isnull().sum().any())
# dfinstagg.to_csv(ROOT + 'installments_payments_preprocessed.csv')

""" EDA - previous_application """
# summary(previous_app)
# missing_data(previous_app)  # 99% of 'RATE_INTEREST_PRIMARY' and 'RATE_INTEREST_PRIVILEGED' are null
# previous_app.drop(['RATE_INTEREST_PRIMARY','RATE_INTEREST_PRIVILEGED'], axis=1, inplace=True)
#
# previous_app['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
# previous_app['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
# previous_app['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
# previous_app['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
# previous_app['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
#
# # New features
# previous_app['APPLICATION_CREDIT_DIFF'] = previous_app['AMT_APPLICATION'] - previous_app['AMT_CREDIT']
# previous_app['APPLICATION_CREDIT_RATIO'] = previous_app['AMT_APPLICATION'] / previous_app['AMT_CREDIT']
# previous_app['CREDIT_TO_ANNUITY_RATIO'] = previous_app['AMT_CREDIT']/previous_app['AMT_ANNUITY']
# previous_app['DOWN_PAYMENT_TO_CREDIT'] = previous_app['AMT_DOWN_PAYMENT'] / previous_app['AMT_CREDIT']
#
# # Null values
# categoric_nulls = [
#     'NAME_TYPE_SUITE',
#     'PRODUCT_COMBINATION',
#     'NFLAG_INSURED_ON_APPROVAL',
# ]
# for col in categoric_nulls:
#     missing_ratio = previous_app[col].isnull().mean()
#
#     if missing_ratio > 0.4:
#         previous_app[col] = previous_app[col].fillna('Missing')
#
#     else:
#         mode_value = previous_app[col].mode()[0]
#         previous_app[col] = previous_app[col].fillna(mode_value)
#
# dfpreapp = convert_types(previous_app, print_info=True)
# dfpreapp_num = aggregate_numerical_features(dfpreapp, 'SK_ID_CURR', 'preapp')
# dfpreapp_categ = agg_categorical(dfpreapp, 'SK_ID_CURR', 'preapp')
#
# """ Final & Save """
# dfpreapplast = pd.merge(dfpreapp_categ, dfpreapp_num, how='left', on="SK_ID_CURR")
# print("Any Null Value dfpreapplast?: ", dfpreapplast.isnull().sum().any())
# dfpreapplast.to_csv(ROOT + 'previous_application_preprocessed.csv')

""" EDA - bureau_balance """
# summary(bureau)
#
# counts_months = (bureau_balance.groupby('SK_ID_BUREAU', as_index=False)['MONTHS_BALANCE']
#                  .count().rename(columns={'MONTHS_BALANCE': 'counts_months'}))
#
# bureau_balance_1 = count_categorical(bureau_balance, 'SK_ID_BUREAU', 'BB')
# bureau_balance_2 = pd.merge(bureau_balance_1, counts_months, on="SK_ID_BUREAU")
# bureau_balance_2['status_rate'] = (bureau_balance_2['BB_STATUS_5_count_norm'] +
#                                    bureau_balance_2['BB_STATUS_4_count_norm'] +
#                                    bureau_balance_2['BB_STATUS_3_count_norm'] +
#                                    bureau_balance_2['BB_STATUS_2_count_norm'] +
#                                    bureau_balance_2['BB_STATUS_1_count_norm'])
# bureau_balance_ready = convert_types(bureau_balance_2, print_info=False)
#
# """ Final & Save """
# bureau_balance_ready.to_csv(ROOT + 'bureau_balance_preprocessed.csv')

""" EDA - bureau """
# summary(bureau)
# bureau = pd.merge(bureau, bureau_balance_ready, how='left', on="SK_ID_BUREAU")
# bureau = bureau.drop(['CREDIT_CURRENCY'], axis=1)
#
# # Create new features
# bureau['duration'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_CREDIT']
# bureau['montly_debt'] = bureau['AMT_CREDIT_SUM'] / bureau['duration'] * 30
# bureau['montly_debt_2'] = bureau['AMT_CREDIT_SUM'] / bureau['counts_months']
# bureau['montly_debt_3'] = bureau['AMT_CREDIT_SUM_DEBT'] / bureau['duration'] * 30
# bureau['montly_debt_4'] = bureau['AMT_CREDIT_SUM_DEBT'] / bureau['counts_months']
# bureau['montly_debt_5'] = np.where(bureau['CREDIT_ACTIVE'] == 'Active',
#                                    bureau['AMT_CREDIT_SUM'] / bureau['counts_months'],
#                                    0)
# bureau['montly_debt_6'] = np.where(bureau['CREDIT_ACTIVE'] == 'Active',
#                                    bureau['AMT_CREDIT_SUM'] / bureau['duration'] * 30,
#                                    0)
# bureau['montly_debt_7'] = np.where(bureau['BB_STATUS_C_count'] < 1,
#                                    bureau['AMT_CREDIT_SUM'] / bureau['duration'] * 30,
#                                    0)
# bureau['montly_debt_8'] = np.where(bureau['BB_STATUS_C_count'] < 1,
#                                    bureau['AMT_CREDIT_SUM'] / bureau['counts_months'],
#                                    0)
#
# bureau_1 = count_categorical(bureau, 'SK_ID_CURR', 'b')
# bureau_2 = aggregate_numerical_features(bureau, 'SK_ID_CURR', 'b')
# bureau_3 = pd.merge(bureau_1, bureau_2, how='left', on="SK_ID_CURR")
#
# f10 = (bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU']
#        .count().rename(columns={'SK_ID_BUREAU': 'previous_loan_counts'}))
# bureau_ready = pd.merge(bureau_3, f10, how='left', on="SK_ID_CURR")
#
# bureau_ready = convert_types(bureau_ready, print_info=True)
#
# """ Final & Save """
# bureau_ready.to_csv(ROOT + 'bureau_preprocessed.csv')

""" EDA - pos_cash """
# summary(pos_cash)
# missing_data(pos_cash)
#
# no_late_payments = (pos_cash[pos_cash['SK_DPD'] > 0]
#                     .groupby(['SK_ID_PREV']).size().reset_index(name='number_late_payments'))
#
# late_pay = pos_cash.groupby('SK_ID_PREV', as_index=False)['SK_DPD'].aggregate(['mean', 'max', "sum"])
# late_pay['SK_DPD_mean'] = late_pay['mean']
# late_pay['SK_DPD_max'] = late_pay['max']
# late_pay['SK_DPD_sum'] = late_pay['sum']
# del late_pay['mean']
# del late_pay['max']
# del late_pay['sum']
#
# pos_cash_lastmonth = pos_cash.loc[pos_cash.groupby('SK_ID_PREV')['MONTHS_BALANCE'].idxmax()]
# num_prev_loan = pos_cash_lastmonth[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
# pos_cash_lastmonth['pos_prev_count'] = pos_cash_lastmonth['SK_ID_CURR'].map(num_prev_loan['SK_ID_PREV'])
#
# pos_cash_lastmonth = pd.get_dummies(pos_cash_lastmonth, prefix='pos_name_contract')
#
# pos_cash_lastmonth['CNT_INSTALMENT_ACTIVE'] = (pos_cash_lastmonth['CNT_INSTALMENT'] *
#                                                pos_cash_lastmonth['pos_name_contract_Active'])
#
# pos_cash_lastmonth = pos_cash_lastmonth.merge(no_late_payments, on=['SK_ID_PREV'], how='left').fillna(0)
# pos_cash_lastmonth = pos_cash_lastmonth.merge(late_pay, on=['SK_ID_PREV'], how='left')
# pos_cash_lastmonth['Ratio_future_instal_to_total_instalments'] = pos_cash_lastmonth['CNT_INSTALMENT_FUTURE'] / \
#                                                                  pos_cash_lastmonth['CNT_INSTALMENT_ACTIVE']
#
# pos_cash_ready = pos_cash_lastmonth.groupby("SK_ID_CURR")
# function_dict = {"MONTHS_BALANCE": "mean",
#                  "CNT_INSTALMENT": "sum",
#                  "CNT_INSTALMENT_FUTURE": "sum",
#                  "CNT_INSTALMENT_ACTIVE": "sum",
#                  "Ratio_future_instal_to_total_instalments": "sum",
#                  "SK_DPD": "sum",
#                  "SK_DPD_mean": "mean",
#                  "SK_DPD_max": "max",
#                  "SK_DPD_sum": "mean",
#                  "number_late_payments": "sum",
#                  "SK_DPD_DEF": "sum",
#                  "pos_name_contract_Active": "sum",
#                  "pos_name_contract_Amortized debt": "sum",
#                  "pos_name_contract_Approved": "sum",
#                  "pos_name_contract_Canceled": "sum",
#                  "pos_name_contract_Completed": "sum",
#                  "pos_name_contract_Demand": "sum",
#                  "pos_name_contract_Returned to the store": "sum",
#                  "pos_name_contract_Signed": "sum",
#                  "pos_prev_count": "mean"}
# pos_cash_ready = pos_cash_ready.aggregate(function_dict)
# pos_cash_ready.reset_index(inplace=True)
#
# """ Final & Save """
# pos_cash_ready.to_csv(ROOT + 'pos_cash_preprocessed.csv')

""" EDA - credit_card_balance """
# summary(credit_card)
# missing_data(credit_card)
#
# counts_less_than_min_payments = (
#     credit_card[credit_card['AMT_PAYMENT_TOTAL_CURRENT'] / credit_card['AMT_INST_MIN_REGULARITY'] < 1]
#     .groupby(['SK_ID_PREV']).size().reset_index(name='Counts_less_than_min_Payments'))
#
# counts_debt_over_limit = (credit_card[credit_card['AMT_BALANCE'] / credit_card['AMT_CREDIT_LIMIT_ACTUAL'] > 1]
#                           .groupby(['SK_ID_PREV']).size().reset_index(name='Counts_amount_over_Limit'))
#
# CCB = credit_card[0:]
# grp = CCB.groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].nunique().reset_index().rename(index=str,
#                                                                                   columns={'SK_ID_PREV': 'NO_LOANS'})
# CCB = CCB.merge(grp, on=['SK_ID_CURR'], how='left')
#
# grp = (CCB.groupby(by=['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].max().reset_index()
#        .rename(index=str, columns={'CNT_INSTALMENT_MATURE_CUM': 'NO_INSTALMENTS'}))
# grp1 = (grp.groupby(by=['SK_ID_CURR'])['NO_INSTALMENTS'].sum().reset_index()
#         .rename(index=str, columns={'NO_INSTALMENTS': 'TOTAL_INSTALMENTS'}))
# CCB = CCB.merge(grp1, on=['SK_ID_CURR'], how='left')
#
# CCB['INSTALLMENTS_PER_LOAN'] = (CCB['TOTAL_INSTALMENTS'] / CCB['NO_LOANS']).astype('uint32')
# del CCB['TOTAL_INSTALMENTS']
# del CCB['NO_LOANS']
#
# counts_late_payments = CCB[CCB['SK_DPD'] > 0].groupby(['SK_ID_PREV']).size().reset_index(name='Counts_late_Payments')
#
# grp = CCB.groupby(by=['SK_ID_CURR'])['SK_DPD'].mean().reset_index().rename(index=str, columns={'SK_DPD': 'AVG_DPD'})
# CCB = CCB.merge(grp, on=['SK_ID_CURR'], how='left')
#
# CCB_lastmonth = CCB.loc[CCB.groupby('SK_ID_PREV')['MONTHS_BALANCE'].idxmax()]
#
# # Feature 4 at the time of application,  for active credits, ratio of Balance (AMT_BALANCE)
# # to total  actual limit (AMT_CREDIT_LIMIT_ACTUAL), is actual limit exceeded or not
# CCB_lastmonth['CREDIT_USAGE'] = CCB_lastmonth['AMT_BALANCE'] / CCB_lastmonth['AMT_CREDIT_LIMIT_ACTUAL']
#
# # Feature5 Dummies for "NAME_CONTRACT_STATUS"
# CCB_lastmonth = pd.get_dummies(CCB_lastmonth, prefix='NAME_CONTRACT_STATUS')
#
# # Merge the Feature 1 formed before, with main data
# # ---Number of payments less than min payments for each credit card (SK_ID_PREV),
# CCB_lastmonth = CCB_lastmonth.merge(counts_less_than_min_payments, on=['SK_ID_PREV'], how='left').fillna(0)
#
# # Merge the Feature 2 formed before, with main data
# # ---does MONTHS_BALANCE exceed AMT_CREDIT_LIMIT_ACTUAL  (SK_ID_PREV), total number of exceeding per loan,
# CCB_lastmonth = CCB_lastmonth.merge(counts_debt_over_limit, on=['SK_ID_PREV'], how='left').fillna(0)
# CCB_lastmonth['ratio_over_limit_usage'] = CCB_lastmonth['Counts_amount_over_Limit'] / CCB_lastmonth[
#     'INSTALLMENTS_PER_LOAN']
#
# # Feature , Feature to calculate number of times Days Past Due occurred, it will be merged to main data
# CCB_lastmonth = CCB_lastmonth.merge(counts_late_payments, on=['SK_ID_PREV'], how='left').fillna(0)
#
# CCB_ready = CCB_lastmonth.groupby("SK_ID_CURR")
#
# function_dict = {"MONTHS_BALANCE": "mean",
#                  "AMT_BALANCE": "sum",
#                  "AMT_CREDIT_LIMIT_ACTUAL": "sum",
#                  "CREDIT_USAGE": "sum",
#                  "Counts_amount_over_Limit": "sum",
#                  "ratio_over_limit_usage": "mean",
#                  "AMT_DRAWINGS_ATM_CURRENT": "sum",
#                  "AMT_DRAWINGS_CURRENT": "sum",
#                  "AMT_DRAWINGS_OTHER_CURRENT": "sum",
#                  "AMT_DRAWINGS_POS_CURRENT": "sum",
#                  "Counts_less_than_min_Payments": "sum",
#                  "AMT_INST_MIN_REGULARITY": "sum",
#                  "AMT_PAYMENT_CURRENT": "sum",
#                  "AMT_PAYMENT_TOTAL_CURRENT": "sum",
#                  "AMT_RECEIVABLE_PRINCIPAL": "sum",
#                  "AMT_RECIVABLE": "sum",
#                  "AMT_TOTAL_RECEIVABLE": "sum",
#                  "CNT_DRAWINGS_ATM_CURRENT": "sum",
#                  "CNT_DRAWINGS_CURRENT": "sum",
#                  "CNT_DRAWINGS_OTHER_CURRENT": "sum",
#                  "CNT_DRAWINGS_POS_CURRENT": "sum",
#                  "CNT_INSTALMENT_MATURE_CUM": "sum",
#                  "SK_DPD": "sum",
#                  "SK_DPD_DEF": "sum",
#                  "Counts_late_Payments": "sum",
#                  "AVG_DPD": "mean",
#                  "INSTALLMENTS_PER_LOAN": "sum",
#                  "NAME_CONTRACT_STATUS_Active": "sum",
#                  "NAME_CONTRACT_STATUS_Completed": "sum",
#                  "NAME_CONTRACT_STATUS_Demand": "sum",
#                  "NAME_CONTRACT_STATUS_Signed": "sum", }
#
# CCB_ready = CCB_ready.aggregate(function_dict)
# CCB_ready.reset_index(inplace=True)
#
# """ Final & Save """
# CCB_ready.to_csv(ROOT + 'credit_card_balance_preprocessed.csv')

""" MERGING """
dftrain = pd.read_csv(ROOT + 'app_train_preprocessed.csv')
bureau_ready = pd.read_csv(ROOT + 'bureau_preprocessed.csv')
pos_cash_ready = pd.read_csv(ROOT + 'pos_cash_preprocessed.csv')
CCB_ready = pd.read_csv(ROOT + 'credit_card_balance_preprocessed.csv')
dfpreapplast = pd.read_csv(ROOT + 'previous_application_preprocessed.csv')
dfinstagg = pd.read_csv(ROOT + 'installments_payments_preprocessed.csv')

for idx, t in enumerate([dftrain, bureau_ready, pos_cash_ready, CCB_ready, dfpreapplast, dfinstagg]):
    print(idx)
    for i in list(t.columns):
        if i.__contains__("Unnamed"):
            print(i, " is containing")
            t.drop(columns=[i], axis=1, inplace=True)

train_ready = pd.merge(dftrain, bureau_ready, how='left', on="SK_ID_CURR")
gc.collect()
train_ready = pd.merge(train_ready, pos_cash_ready, how='left', on="SK_ID_CURR")
gc.collect()
train_ready = pd.merge(train_ready, CCB_ready, how='left', on="SK_ID_CURR")
gc.collect()
train_ready = pd.merge(train_ready, dfpreapplast, how='left', on="SK_ID_CURR")
gc.collect()
train_ready = pd.merge(train_ready, dfinstagg, how='left', on="SK_ID_CURR")
gc.collect()

train_ready['credit_power1'] = train_ready['AMT_INCOME_TOTAL'] / train_ready['b_montly_debt_sum']
train_ready['credit_power2'] = train_ready['AMT_INCOME_TOTAL'] / train_ready['b_montly_debt_2_sum']
train_ready['credit_power3'] = train_ready['AMT_INCOME_TOTAL'] / train_ready['b_montly_debt_3_sum']
train_ready['credit_power4'] = train_ready['AMT_INCOME_TOTAL'] / train_ready['b_montly_debt_4_sum']
train_ready['credit_power5'] = train_ready['AMT_INCOME_TOTAL'] / train_ready['b_montly_debt_5_sum']
train_ready['credit_power6'] = train_ready['AMT_INCOME_TOTAL'] / train_ready['b_montly_debt_6_sum']
train_ready['credit_power7'] = train_ready['AMT_INCOME_TOTAL'] / train_ready['b_montly_debt_7_sum']
train_ready['credit_power8'] = train_ready['AMT_INCOME_TOTAL'] / train_ready['b_montly_debt_8_sum']

train_ready['CHILDREN_TO_FAMILY'] = train_ready['CNT_CHILDREN'] / train_ready['CNT_FAM_MEMBERS']
train_ready['CREDIT_TO_INCOME'] = train_ready['AMT_CREDIT'] / train_ready['AMT_INCOME_TOTAL']
train_ready['CREDIT_TERM'] = train_ready['AMT_CREDIT'] / train_ready['AMT_ANNUITY']
train_ready['ANNUITY_TO_INCOME'] = train_ready['AMT_ANNUITY'] / train_ready['AMT_INCOME_TOTAL']
train_ready['GOODS_PRICE_TO_CREDIT'] = train_ready['AMT_GOODS_PRICE'] / train_ready['AMT_CREDIT']
train_ready['INCOME_PER_CHILD'] = train_ready['AMT_INCOME_TOTAL'] / train_ready['CNT_CHILDREN']
train_ready['RATE_EMPLOYED'] = train_ready['DAYS_EMPLOYED'] / train_ready['DAYS_BIRTH']
train_ready['RATE_30_SOCIAL'] = train_ready['DEF_30_CNT_SOCIAL_CIRCLE'] / train_ready['OBS_30_CNT_SOCIAL_CIRCLE']
train_ready['RATE_60_SOCIAL'] = train_ready['DEF_60_CNT_SOCIAL_CIRCLE'] / train_ready['OBS_60_CNT_SOCIAL_CIRCLE']
# train_ready['YEARS_EMPLOYED'] = train_ready['DAYS_EMPLOYED'] / -365
# train_ready['YEARS_BIRTH'] = train_ready['DAYS_BIRTH'] / -365

train_ready['INTEREST'] = train_ready['preapp_CNT_PAYMENT_sum'] * train_ready['AMT_ANNUITY'] - train_ready['AMT_CREDIT']
train_ready['INTEREST_RATE'] = 2 * 12 * train_ready['INTEREST'] / (
        train_ready['AMT_CREDIT'] * (train_ready['preapp_CNT_PAYMENT_sum'] + 1))
train_ready['INTEREST_SHARE'] = train_ready['INTEREST'] / train_ready['AMT_CREDIT']

train_ready['income_balance'] = train_ready['AMT_INCOME_TOTAL'] / train_ready['AMT_BALANCE']
train_ready['income_regularity'] = train_ready['AMT_INCOME_TOTAL'] / train_ready['AMT_INST_MIN_REGULARITY']
train_ready['balance_regularity'] = train_ready['income_balance'] / train_ready['income_regularity']

train_ready = convert_types(train_ready, print_info=False)

""" Final & Save """
train_ready.to_csv(ROOT + 'mergetrain.csv', index=False)
