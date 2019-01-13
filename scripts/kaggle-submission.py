'''
See accompanying notebook with Exploratory Data Analysis:
https://www.kaggle.com/alhankeser/beginner-eda-and-data-cleaning
'''

# External libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import explained_variance_score
import scipy.stats as stats
import math
import time
import traceback
import warnings
# import operator
# import functools
# import sys

# Options
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)
warnings.filterwarnings(action="ignore")


class Explore:

    def get_dtype(cls, include_type=[], exclude_type=[]):
        df = cls.get_df('train')
        df.drop(columns=[cls.target_col], inplace=True)
        return df.select_dtypes(include=include_type, exclude=exclude_type)

    def get_non_numeric(cls):
        return cls.get_dtype(exclude_type=['float64', 'int', 'float32'])

    def get_numeric(cls):
        return cls.get_dtype(exclude_type=['object'])

    def get_categorical(cls, as_df=False):
        return cls.get_dtype(include_type=['object'])

    def get_correlations(cls, method='spearman'):
        df = cls.get_df('train')
        corr_mat = df.corr(method=method)
        corr_mat.sort_values(cls.target_col, inplace=True)
        corr_mat.drop(cls.target_col, inplace=True)
        return corr_mat[[cls.target_col]]

    def get_skewed_features(cls, df, features, skew_threshold=0.4):
        feat_skew = pd.DataFrame(
                    {'skew': df[features].apply(lambda x: stats.skew(x))})
        skewed = feat_skew[abs(feat_skew['skew']) > skew_threshold].index
        return skewed.values

    def show_boxplot(cls, x, y, **kwargs):
        sns.boxplot(x=x, y=y)
        x = plt.xticks(rotation=90)

    def plot_categorical(cls, df, cols):
        categorical = pd.melt(df, id_vars=['SalePrice'],
                              value_vars=cols)
        grouped = categorical.groupby(['value', 'variable'],
                                      as_index=False)['SalePrice']\
            .median().rename(columns={'SalePrice': 'SalePrice_Median'})
        categorical = pd.merge(categorical, grouped, how='left',
                               on=['variable', 'value'])\
            .sort_values('SalePrice_Median')
        facet_grid = sns.FacetGrid(categorical, col="variable",
                                   col_wrap=3, size=5,
                                   sharex=False, sharey=False,)
        facet_grid = facet_grid.map(cls.show_boxplot, "value", "SalePrice")
        plt.savefig('boxplots.png')


class Clean:

    def remove_outliers(cls, df):
        if df.name == 'train':
            # GrLivArea (1299 & 524)
            df.drop(df[(df['GrLivArea'] > 4000) &
                    (df[cls.target_col] < 300000)].index,
                    inplace=True)
        return df

    def fill_by_type(cls, x, col):
        if pd.isna(x):
            if col.dtype == 'object':
                return ''
            return 0
        return x

    def fill_na(cls, df):
        for col in df.columns:
            df[col] = df[col].apply(lambda x: cls.fill_by_type(x, df[col]))
        return df

    def get_encoding_lookup(cls, cols):
        df = cls.get_df('train')
        target = cls.target_col
        suffix = '_E'
        result = pd.DataFrame()
        for cat_feat in cols:
            cat_feat_target = df[[cat_feat, target]].groupby(cat_feat)
            cat_feat_encoded_name = cat_feat + suffix
            order = pd.DataFrame()
            order['val'] = df[cat_feat].unique()
            order.index = order.val
            order.drop(columns=['val'], inplace=True)
            order[target + '_median'] = cat_feat_target[[target]].median()
            order['feature'] = cat_feat
            order['encoded_name'] = cat_feat_encoded_name
            order = order.sort_values(target + '_median')
            order['num_val'] = range(1, len(order)+1)
            result = result.append(order)
        result.reset_index(inplace=True)
        return result

    def get_scaled_categorical(cls, encoding_lookup):
        scaled = encoding_lookup.copy()
        target = cls.target_col
        for feature in scaled['feature'].unique():
            values = scaled[scaled['feature'] == feature]['num_val'].values
            medians = scaled[
                    scaled['feature'] == feature][target + '_median'].values
            for median in medians:
                scaled_value = ((values.min() + 1) *
                                (median / medians.min()))-1
                scaled.loc[(scaled['feature'] == feature) &
                           (scaled[target + '_median'] == median),
                           'num_val'] = scaled_value
        return scaled

    def encode_with_lookup(cls, df, encoding_lookup):
        for encoded_index, encoded_row in encoding_lookup.iterrows():
            feature = encoded_row['feature']
            encoded_name = encoded_row['encoded_name']
            value = encoded_row['val']
            encoded_value = encoded_row['num_val']
            df.loc[df[feature] == value, encoded_name] = encoded_value
        return df

    def encode_onehot(cls, df, cols):
        df = pd.concat([df, pd.get_dummies(df[cols], drop_first=True)], axis=1)
        print(df.head(2))
        return df

    def encode_categorical(cls, df, cols=[], method='one_hot'):
        if len(cols) == 0:
            cols = cls.get_categorical().columns.values
        if method == 'target_median':
            encoding_lookup = cls.get_encoding_lookup(cols)
            encoding_lookup = cls.get_scaled_categorical(encoding_lookup)
            df = cls.encode_with_lookup(df, encoding_lookup)
        if method == 'one_hot':
            df = cls.encode_onehot(df, cols)
        df.drop(cols, axis=1, inplace=True)
        return df

    def fix_zero_infinity(cls, x):
        if (x == 0) or math.isinf(x):
            return 0
        return x

    def normalize_features(cls, df, cols=[]):
        if len(cols) == 0:
            cols = cls.get_numeric().columns.values
        for col in cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x:
                                        np.log1p(x).astype('float64'))
                df[col] = df[col].apply(lambda x: cls.fix_zero_infinity(x))
        return df

    def scale_quant_features(cls, df, cols):
        scaler = StandardScaler()
        scaler.fit(df[cols])
        scaled = scaler.transform(df[cols])
        for i, col in enumerate(cols):
            df[col] = scaled[:, i]
        return df

    def drop_ignore(cls, df):
        for col in cls.ignore:
            try:
                df.drop(col, axis=1, inplace=True)
            except Exception:
                pass
        return df

    def drop_low_corr(cls, df, threshold=0.12):
        to_drop = pd.DataFrame(columns=['drop'])
        corr_mat = cls.get_correlations()
        target = cls.target_col
        to_drop['drop'] = corr_mat[(abs(corr_mat[target]) <= threshold)].index
        df.drop(to_drop['drop'], axis=1, inplace=True)
        return df


class Engineer:

    def group_slope(cls, x):
        if x in ['Sev', 'Mod']:
            return 'Slope'
        return x

    def land_slope(cls, df):
        df['LandSlope'] = df['LandSlope'].apply(lambda x: cls.group_slope(x))
        return df

    def group_lot(cls, x):
        if x in ['Inside', 'Corner', 'FR2']:
            return 'Okay'
        return x

    def lot_config(cls, df):
        df['LotConfig'] = df['LotConfig']\
                          .apply(lambda x: cls.group_lot(x))
        return df

    def group_paved(cls, x):
        if x in ['PaveGrvlN', 'PaveN']:
            return 'LowPave'
        if x in ['GrvlY', 'PaveGrvlY', 'PaveP', 'PaveGrvlP']:
            return 'MedPave'
        if x in ['PaveY', 'PavePaveY', 'PavePaveN']:
            return 'HighPave'
        return x

    def paved(cls, df):
        df['Street__Alley__PavedDrive'] = df['Street__Alley__PavedDrive']\
                                          .apply(lambda x: cls.group_paved(x))
        return df

    def group_mszoning(cls, x):
        if (x == 'RM') or (x == 'RH'):
            return 'RMRH'
        if (x == 'RL') or (x == 'FV'):
            return 'RLFV'
        return x

    def mszoning(cls, df):
        df['MSZoning'] = df['MSZoning'].apply(lambda x: cls.group_mszoning(x))
        return df

    def group_electrical(cls, x):
        if (x == 'Mix') or (x == 'FuseP'):
            return 'Poor'
        if (x == 'FuseF') or (x == 'FuseA'):
            return 'Okay'
        else:
            return 'Good'

    def electrical_quality(cls, df):
        df['Electrical'] = df['Electrical'].apply(
                           lambda x: cls.group_electrical(x))
        return df

    def is_positive_subclass(cls, df):
        df['MSSubClass'] = df['MSSubClass'].apply(
                           lambda x: bool((x == 60) or (x == 120)))
        return df

    def has_deduction(cls, df):
        df['Functional'] = (df['Functional'] != 'Typ').astype(bool)
        # df.drop('Functional', axis=1, inplace=True)
        return df

    def convert_to_string(cls, df, cols):
        for col in cols:
            df[col] = df[col].apply(lambda x: '_' + str(x))
        return df

    def is_regular(cls, x):
        if x in ['IR1', 'IR2']:
            return 'SemiReg'
        return x

    def lot_shape(cls, df):
        df['LotShape'] = df['LotShape'].apply(lambda x: cls.is_regular(x))
        # df.drop('LotShape', axis=1, inplace=True)
        return df

    def group_pool(cls, x):
        if x in ['Ex', 'Gd']:
            return 'Good'
        return x

    def pool(cls, df):
        df['PoolQC'] = df['PoolQC'].apply(lambda x: cls.group_pool(x))
        # df.drop('PoolQC', axis=1, inplace=True)
        return df

    def bath_porch_sf(cls, df):
        # Total SF for bathroom
        df['TotalBath'] = df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']) + \
            df['FullBath'] + (0.5 * df['HalfBath'])

        # Total SF for porch
        df['AllPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + \
            df['3SsnPorch'] + df['ScreenPorch']

        # Drop original columns
        df.drop(['BsmtFullBath', 'FullBath', 'HalfBath',
                'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch'],
                inplace=True, axis=1)
        return df

    def house_remodel_age(cls, df):
        # if no remodeling or additions))
        df['Is_Remodeled'] = (df['YearRemodAdd'] !=
                              df['YearBuilt']).astype(bool)

        # add feature about the age of the house when sold
        df['Age'] = df['YrSold'] - df['YearBuilt']

        df['Garage_Age'] = df['YrSold'] - df['GarageYrBlt']

        # add flag if house was sold 2 years or less after it was built
        df['Is_New_House'] = (df['YrSold'] - df['YearBuilt'] <= 2).astype(bool)

        # add flag is remodel was recent (i.e. within 2 years of the sale)
        df['Is_Recent_Remodel'] = (df['YrSold'] -
                                   df['YearRemodAdd'] <= 2).astype(bool)

        # drop the original columns
        df.drop(['YearRemodAdd', 'YearBuilt', 'GarageYrBlt'],
                axis=1, inplace=True)
        return df

    def garage_age(cls, df):
        # if no remodeling or additions))
        df['Is_Remodeled'] = (df['YearRemodAdd'] !=
                              df['YearBuilt']).astype(bool)

        # add feature about the age of the house when sold
        df['Age'] = df['YrSold'] - df['YearBuilt']

        # add flag if house was sold 2 years or less after it was built
        df['Is_New_House'] = (df['YrSold'] - df['YearBuilt'] <= 2).astype(bool)

        # add flag is remodel was recent (i.e. within 2 years of the sale)
        df['Is_Recent_Remodel'] = (df['YrSold'] -
                                   df['YearRemodAdd'] <= 2).astype(bool)

        # drop the original columns
        df.drop(['YearRemodAdd', 'YearBuilt'], axis=1, inplace=True)
        return df

    def sum_features(cls, df, col_sum):
        for col_set in col_sum:
            f_name = '__'.join(col_set[:])
            df[f_name] = df[[*col_set]].sum(axis=1)
            df.drop(col_set, axis=1, inplace=True)
        return df

    def multiply_features(cls, df, feature_sets):
        for feature_set in feature_sets:
            # multipled_name = '_x_'.join(feature_set[:])
            # df.drop(feature_set, axis=1, inplace=True)
            pass
        return df


class Model:

    # def cross_validate(cls, model, random_state=0):
    #     train = cls.get_df('train')
    #     X = train.drop(columns=[cls.target_col])
    #     y = train[cls.target_col]
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=0.3,
    #         random_state=random_state)
    #     model = model()
    #     model.fit(X_train, y_train)
    #     X_predictions = model.predict(X_test)
    #     score = math.sqrt(mean_squared_error(y_test, X_predictions))
    #     return (model, score)

    def grid_search(cls, model, parameters):
        train, test = cls.get_dfs()
        X = train.drop(columns=[cls.target_col])
        y = train[cls.target_col]
        model = GridSearchCV(model(), parameters, cv=10,
                             scoring='neg_mean_squared_error')
        model.fit(X, y)
        return model

    def predict(cls, model):
        test = cls.get_df('test')
        predictions = model.predict(test)
        return predictions

    def save_predictions(cls, predictions):
        now = str(time.time()).split('.')[0]
        df = cls.get_df('test', False, True)
        target = cls.target_col
        df[target] = predictions
        df[target] = df[target].apply(lambda x: np.expm1(x))
        df[[df.columns[0], target]].to_csv('submit-' + now + '.csv',
                                           index=False)


class Data(Explore, Clean, Engineer, Model):

    def __init__(self, train_csv, test_csv, target='', ignore=[], col_sum=[]):
        '''Create pandas DataFrame objects for train and test data.

        Positional arguments:
        train_csv -- relative path to training data in csv format.
        test_csv -- relative path to test data in csv format.

        Keyword arguments:
        target -- target feature column name in training data.
        ignore -- columns names in list to ignore during analyses.
        '''
        self.__train = pd.read_csv(train_csv)
        self.__test = pd.read_csv(test_csv)
        self.__train.name, self.__test.name = self.get_df_names()
        self.target_col = target
        self.ignore = ignore
        self.col_sum = col_sum
        self.__original = False
        self.__log = False
        self.check_in()
        self.debug = False

    def __str__(cls):
        train_columns = 'Train: \n"' + '", "'.join(cls.__train.head(2)) + '"\n'
        test_columns = 'Test: \n"' + '", "'.join(cls.__test.head(2)) + '"\n'
        return train_columns + test_columns

    def get_df_names(cls):
        return ('train', 'test')

    def get_dfs(cls, ignore=False, originals=False):
        train, test = (cls.__train.copy(),
                       cls.__test.copy())
        if originals:
            train, test = (cls.__original)
        if ignore:
            train, test = (train.drop(columns=cls.ignore),
                           test.drop(columns=cls.ignore))
        train.name, test.name = cls.get_df_names()
        return (train, test)

    def get_df(cls, name, ignore=False, original=False):
        train, test = cls.get_dfs(ignore, original)
        if name == 'train':
            return train
        if name == 'test':
            return test

    def log(cls, entry=False, status=False):
        if cls.__log is False:
            cls.__log = pd.DataFrame(columns=['entry', 'status'])
        log_entry = pd.DataFrame({'entry': entry, 'status': status}, index=[0])
        cls.__log = cls.__log.append(log_entry, ignore_index=True)
        if status == 'Fail':
            cls.rollback()
        else:
            cls.check_out()
            if cls.debug:
                cls.print_log()

    def print_log(cls):
        print(cls.__log)

    def check_in(cls):
        cls.__current = cls.get_dfs()
        if cls.__original is False:
            cls.__original = cls.__current

    def check_out(cls):
        cls.__previous = cls.__current
        cls.__train.name, cls.__test.name = cls.get_df_names()

    def rollback(cls):
        try:
            cls.__train, cls.__test = cls.__previous
            status = 'Success - To Previous'
        except Exception:
            cls.__train, cls.__test = cls.__original
            status = 'Success - To Original'
        cls.log('rollback', status)

    def reset(cls):
        cls.__train, cls.__test = cls.__original
        cls.log('reset', 'Success')

    def update_dfs(cls, train, test):
        train.name, test.name = cls.get_df_names()
        cls.__train = train
        cls.__test = test

    def mutate(cls, mutation, *args):
        '''Make changes to both train and test DataFrames.
        Positional arguments:
        mutation -- function to pass both train and test DataFrames to.
        *args -- arguments to pass to the function, following each DataFrame.

        Example usage:
        def multiply_column_values(df, col_name, times=10):
            #do magic...

        Data.mutate(multiply_column_values, 'Id', 2)
        '''
        cls.check_in()
        try:
            train = mutation(cls.get_df('train'), *args)
            test = mutation(cls.get_df('test'), *args)
            cls.update_dfs(train, test)
            status = 'Success'
        except Exception:
            print(traceback.print_exc())
            status = 'Fail'
        cls.log(mutation.__name__, status)


def run(d, model, parameters):
    mutate = d.mutate
    mutate(d.fill_na)
    mutate(d.remove_outliers)

    mutate(d.lot_config)
    mutate(d.lot_shape)
    mutate(d.land_slope)

    mutate(d.house_remodel_age)

    mutate(d.convert_to_string, ['MSSubClass', 'YrSold', 'MoSold'])
    # Feature Engineering
    mutate(d.sum_features, d.col_sum)
    mutate(d.bath_porch_sf)
    mutate(d.pool)
    mutate(d.has_deduction)
    # mutate(d.is_positive_subclass)
    mutate(d.electrical_quality)
    # mutate(d.mszoning)
    # mutate(d.paved)

    # Show categorical facetgrid w/ boxplots
    # categorical = d.get_non_numeric().columns.values
    # train = d.get_df('train')
    # d.plot_categorical(train, categorical)
    numeric_cols = d.get_numeric().columns.values
    mutate(d.scale_quant_features, numeric_cols)
    mutate(d.encode_categorical, [], 'target_median')
    mutate(d.normalize_features, [d.target_col])

    # skewed_features = d.get_skewed_features(d.get_df('train'), numeric_cols)
    # mutate(d.normalize_features, skewed_features)
    mutate(d.drop_low_corr)
    mutate(d.drop_ignore)
    mutate(d.fill_na)

    # print(d.get_df('train').corr()['SalePrice'].sort_values())
    model = d.grid_search(model, parameters)
    print(round(model.cv_results_['mean_test_score'][0], 7))
    # print(model.cv_results_['mean_test_nmse'])
    # print(model.cv_results_['mean_train_nmse'])
    predictions = d.predict(model)
    d.print_log()
    return predictions


model = LinearRegression
parameters = {}
# cols_to_ignore = ['Id', 'BedroomAbvGr', 'GarageArea',
#                   'FireplaceQu_E', 'Alley_E', 'MasVnrArea', 'Condition2_E']
cols_to_ignore = ['Id', 'BedroomAbvGr', 'GarageArea',
                  'FireplaceQu_E', 'Alley_E', 'MasVnrArea', 'Condition2_E']
col_sum = [
    # ['LotShape', 'LandContour'],
    # ['Condition1', 'Condition2'],
    # ['BldgType', 'HouseStyle'],
    # ['RoofStyle', 'RoofMatl'],
    # ['Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond'],
    # ['BsmtQual', 'BsmtCond'],
    # ['HeatingQC', 'CentralAir'],
    # ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'],
    # ['Functional', 'LandContour']
    # ['YrSold', 'MoSold'],
    # ['Street', 'Alley', 'PavedDrive']
]
d = Data('./input/train.csv',
         './input/test.csv',
         'SalePrice',
         cols_to_ignore,
         col_sum)
# print(d.get_df('train').columns)
# print(d.get_df('train')[['Functional']].groupby('Functional'))
predictions = run(d, model, parameters)
d.save_predictions(predictions)
# -0.013121
