'''
DISCLAIMER:
Refactor in progress.
'''

# External libraries
import sys
# import numpy as np
import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import StandardScaler
# import scipy.stats as stats
# from scipy.special import boxcox1p
# import operator
# import functools
# import math
# import time
import warnings
warnings.filterwarnings(action="ignore")


class Explore:

    def get_dtype(cls, include_type=[], exclude_type=[]):
        df = cls.get_df('train')
        df.drop(columns=[cls.target_col], inplace=True)
        return df.select_dtypes(include=include_type, exclude=exclude_type)

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


class Data(Explore):

    def __init__(self, train_csv, test_csv, target='', ignore=[]):
        '''Create pandas DataFrame objects for train and test data.

        Positional arguments:
        train_csv -- relative path to training data in csv format.
        test_csv -- relative path to test data in csv format.

        Keyword arguments:
        target -- target feature column name in training data.
        ignore -- columns names in list to ignore during analyses.
        '''
        self.__train = pd.read_csv(train_csv)
        self.__train.name = 'train'
        self.__test = pd.read_csv(test_csv)
        self.__test.name = 'test'
        self.target_col = target
        self.target = self.__train[[self.target_col]]
        self.ignore = ignore
        self.__original = False
        self.__log = False
        self.check_in()
        self.debug = True

    def __str__(cls):
        train_columns = 'Train: \n"' + '", "'.join(cls.__train.head(2)) + '"\n'
        test_columns = 'Test: \n"' + '", "'.join(cls.__test.head(2)) + '"\n'
        return train_columns + test_columns

    def get_dfs(cls, ignore=True):
        train, test = (cls.__train.copy(),
                       cls.__test.copy())
        train, test = (train.drop(columns=cls.ignore),
                       test.drop(columns=cls.ignore))
        train.name, test.name = (cls.__train.name,
                                 cls.__test.name)
        return (train, test)

    def get_df(cls, name, ignore=True):
        train, test = cls.get_dfs(ignore)
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
            print(cls.__log)

    def check_in(cls):
        cls.__current = cls.get_dfs()
        if cls.__original is False:
            cls.__original = cls.__current

    def check_out(cls):
        cls.__previous = cls.__current

    def rollback(cls):
        cls.__train, cls.__test = cls.__previous
        cls.log('rollback', 'Success')

    def reset(cls):
        cls.__train, cls.__test = cls.__original
        cls.log('reset', 'Success')

    def report_exception(cls, ex_type, ex_value, name=False):
        print(ex_type.__name__, ex_value, name)

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
            mutation(cls.__train, *args)
            mutation(cls.__test, *args)
            status = 'Success'
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            cls.report_exception(exc_type, exc_value, mutation.__name__)
            status = 'Fail'
        cls.log(mutation.__name__, status)


d = Data('./input/train.csv', './input/test.csv', 'SalePrice', ['Id'])

numeric = d.get_numeric()
categorical = d.get_categorical()
corrs = d.get_correlations()
print(corrs)


# print(numeric)
# print(len(numeric))

# print(categorical)
# print(len(categorical))

# print(len(d.get_df('train').columns))

'''

    class Explore:

        def run(cls, train_clean, qual_features_encoded, target_feature):
            qual_features_list = cls.qual_features_encoded['encoded_name'].unique()
            correlations = cls.get_correlations(train_clean, target_feature)
            disparity = False # get_qual_feature_disparity(train_clean, qual_features_list, target_feature)
            # effect_size = compare_qual_feature_value_effect(train_clean, qual_features_list, target_feature)
            return (correlations, disparity)

        def get_quant_features(cls, df, target_feature, ignore_features):
            ignore_features = ignore_features.copy()
            features = [f for f in df.columns if df.dtypes[f] != 'object']
            ignore_features.append(target_feature)
            features = list(set(features) - set(ignore_features))
            return features


        def get_qual_features(cls, df, target_feature, ignore_features):
            ignore_features = ignore_features.copy()
            features = [f for f in df.columns if df.dtypes[f] == 'object']
            ignore_features.append(target_feature)
            features = list(set(features) - set(ignore_features))
            return features


        def get_correlations(cls, df, target_feature, method='spearman'):
            correlation_matrix = df.corr(method=method)
            correlation_matrix = correlation_matrix.sort_values(target_feature)
            correlation_matrix = correlation_matrix.drop(target_feature)
            return correlation_matrix[[target_feature]]


        def get_qual_feature_disparity(cls, df, qual_features_list, target_feature):
            anova_df = pd.DataFrame()
            anova_df['feature'] = qual_features_list
            p_values = []
            for col in qual_features_list:
                samples = []
                for unique_val in df[col].unique():
                    sample = df[df[col] == unique_val][target_feature].values
                    samples.append(sample)
                p_value = stats.f_oneway(*samples)[1]
                p_values.append(p_value)
            anova_df['p_value'] = p_values
            anova_df['disparity'] = np.log(1./anova_df['p_value'].values)
            # sort
            return anova_df.sort_values('disparity')

        def compare_qual_feature_value_effect(cls, df, qual_features_list, target_feature):
            target_median = target_feature + '_median'
            result = pd.DataFrame(columns=['feature', 'value', target_median])
            for col in qual_features_list:
                for unique_val in df[col].unique():
                    result = result.append({
                        'feature': col,
                        'value': unique_val, 
                        target_median: df[df[col] == unique_val][target_feature].median()
                    }, ignore_index=True)
            result = result.sort_values(['feature', 'value'])
            return result

    class Clean:

        def run(cls, DATA, CONFIG):
            train_clean = DATA['TRAIN'].copy()
            test_clean = DATA['TEST'].copy()
            normalize_target = CONFIG['options']['normalize_target']
            scale_encoded_qual_features = CONFIG['options']['scale_encoded_qual_features']
            target_feature = DATA['TARGET_FEATURE']
            quant_features = DATA['QUANT_FEATURES']
            normalize_quant_features = CONFIG['options']['normalize_quant_features']
            skew_threshold = CONFIG['options']['skew_threshold']

            # Remove outliers
            train_clean = train_clean.drop(train_clean[(train_clean['GrLivArea']>4000) & (train_clean[target_feature]<300000)].index)
            train_clean = train_clean.drop(train_clean[(train_clean['TotalBsmtSF']>6000)].index)
            train_clean = train_clean.drop(train_clean[(train_clean['1stFlrSF']>4000)].index)
            train_clean = train_clean.drop(train_clean[(train_clean['GarageArea']>1400) & (train_clean[target_feature]<300000)].index)

            # Create qual features encoding lookup table
            qual_features_encoded = cls.create_encoding_lookup(train_clean.fillna(0), DATA['QUAL_FEATURES'], DATA['TARGET_FEATURE'])
            # Get skewed features based on training data
            skewed_features = cls.get_skewed_features(train_clean, quant_features, skew_threshold)
            if scale_encoded_qual_features:
                qual_features_encoded = cls.scale_qual_feature_encoding(qual_features_encoded, DATA['TARGET_FEATURE'])
            if normalize_target:
                train_clean[target_feature] = np.log1p(train_clean[target_feature])
            # Both
            dfs = [train_clean, test_clean]
            result = [qual_features_encoded]
            for df in dfs:
                df = df.fillna(0)
                df = cls.encode_qual_features(df, qual_features_encoded)
                if normalize_quant_features:
                    df = cls.normalize_features(df, skewed_features)
                df = df.fillna(0)
                df = df.drop(columns=DATA['IGNORE_FEATURES'])
                result.append(df)
            return result

        def get_skewed_features(df, features, skew_threshold):
            feature_skew = pd.DataFrame({'skew': df[features].apply(lambda x: stats.skew(x))})
            skewed_features = feature_skew[abs(feature_skew['skew']) > skew_threshold].index
            return skewed_features

        def normalize_features(df, skewed_features):
            # print(is_skewed)
            for feature in skewed_features:
                # if feature in is_skewed:
                # df[feature] = df[feature].apply(lambda x: boxcox1p(x, 0.15))
                # else:
                df[feature] = df[feature].apply(lambda x: np.log1p(x))
            return df

        def create_encoding_lookup(df, qual_features, target_feature, suffix='_E'):
            result = pd.DataFrame([])
            for qual_feature in qual_features:
                order_df = pd.DataFrame()
                order_df['val'] = df[qual_feature].unique()
                order_df.index = order_df.val
                order_df.drop(columns=['val'], inplace=True)
                order_df[target_feature + '_median'] = df[[qual_feature, target_feature]].groupby(qual_feature)[[target_feature]].median()
                qual_feature_encoded_name = qual_feature + suffix
                order_df['feature'] = qual_feature
                order_df['encoded_name'] = qual_feature_encoded_name
                order_df = order_df.sort_values(target_feature + '_median')
                order_df['num_val'] = range(1, len(order_df)+1)
                result = result.append(order_df)
            result.reset_index(inplace=True)
            return result


        def encode_qual_features(df, qual_features_encoded):
            result = df.copy()
            for encoded_index, encoded_row in qual_features_encoded.iterrows():
                feature = encoded_row['feature']
                encoded_name = encoded_row['encoded_name']
                value = encoded_row['val']
                encoded_value = encoded_row['num_val'] 
                result.loc[result[feature] == value, encoded_name] = encoded_value
            result = result.drop(columns=qual_features_encoded['feature'].unique())
            return result

        def scale_qual_feature_encoding(qual_features_encoded, target_feature):
            result = qual_features_encoded.copy()
            for feature in result['feature'].unique():
                values = result[result['feature'] == feature]['num_val'].values
                medians = result[result['feature'] == feature][target_feature + '_median'].values
                for median in medians:
                    # scaled_value_max = len(values) * (median / medians.max())
                    scaled_value = ((values.min() + 1) * (median / medians.min()))-1
                    result.loc[(result['feature'] == feature) & (result[target_feature + '_median'] == median), 'num_val'] = scaled_value
            return result

    class Engineer:

        def drop_features(clean_train, clean_test, target_feature, drop, correlations=False, threshold=False):
            dfs = [clean_train, clean_test]
            features_to_drop = pd.DataFrame(columns=['drop'])
            correlations_to_drop = pd.DataFrame()
            if len(drop) > 0:
                features_to_drop['drop'] = drop
            if len(correlations) > 0 and threshold is not False:
                correlations_to_drop['drop'] = correlations[(correlations[target_feature] <= threshold) & (correlations[target_feature] >= (threshold * -1))].index
            features_to_drop = features_to_drop.merge(correlations_to_drop, how='outer', on='drop')
            dfs[0].drop(columns=features_to_drop['drop'], inplace=True)
            dfs[1].drop(columns=features_to_drop['drop'], inplace=True)
            return dfs

        def sum_features(clean_train, clean_test, feature_sets):
            dfs = [clean_train, clean_test]
            result = pd.DataFrame([])
            for feature_set in feature_sets:
                if len(feature_set) > 2:
                    raise ValueError('Only put 2 vars at a time to sum.') 
                sumd_name = '_+_'.join(feature_set[:])
                dfs[0][sumd_name] = dfs[0][feature_set[0]] + dfs[0][feature_set[1]]
                dfs[0] = dfs[0].drop(columns=feature_set)
                dfs[1][sumd_name] = dfs[1][feature_set[0]] + dfs[1][feature_set[1]]
                dfs[1] = dfs[1].drop(columns=feature_set)
            return dfs

        def multiply_features(clean_train, clean_test, feature_sets):
            dfs = [clean_train, clean_test]
            for feature_set in feature_sets:
                multipled_name = '_x_'.join(feature_set[:])
                dfs[0][multipled_name] = dfs[0][feature_set[0]] * dfs[0][feature_set[1]]
                dfs[0] = dfs[0].drop(columns=feature_set)
                dfs[1][multipled_name] = dfs[1][feature_set[0]] * dfs[1][feature_set[1]]
                dfs[1] = dfs[1].drop(columns=feature_set)
            return dfs

        def make_binary(dfs, target_feature):
            # TODO        
            return True

        def bath_porch_sf(train_clean, test_clean):
            dfs = [train_clean, test_clean]
            result = []
            for df in dfs:
                # total SF for bathroom
                df['TotalBath'] = df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']) + \
                df['FullBath'] + (0.5 * df['HalfBath'])

                # Total SF for porch
                df['AllPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + \
                df['3SsnPorch'] + df['ScreenPorch']
                
                # drop the original columns
                df = df.drop(['BsmtFullBath', 'FullBath', 'HalfBath', 
                            'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch'], axis=1)
                result.append(df)
            return result

        def house_remodel_and_age(train_clean, test_clean):
            dfs = [train_clean, test_clean]
            result = []
            for df in dfs:
                # add flag is house has been remodeled (the year of the remodel is same as construction date 
                # if no remodeling or additions))
                df['is_remodeled'] = (df['YearRemodAdd'] != df['YearBuilt'])

                # add feature about the age of the house when sold
                df['age'] = df['YrSold'] - df['YearBuilt']

                # add flag if house was sold 2 years or less after it was built
                df['is_new_house'] = (df['YrSold'] - df['YearBuilt'] <= 2)

                # add flag is remodel was recent (i.e. within 2 years of the sale)
                df['is_recent_remodel'] = (df['YrSold'] - df['YearRemodAdd'] <= 2)
                
                # drop the original columns
                df = df.drop(['YearRemodAdd', 'YearBuilt'], axis=1)
                result.append(df)
            return result

        def scale_quant_features(train_clean, test_clean, quant_features):
            scaler = StandardScaler()
            scaler.fit(train_clean[quant_features])
            scaled = scaler.transform(train_clean[quant_features])
            for i, col in enumerate(quant_features):
                train_clean[col] = scaled[:,i]
            scaled = scaler.fit_transform(test_clean[quant_features])
            for i, col in enumerate(quant_features):
                test_clean[col] = scaled[:,i]
            return (train_clean, test_clean)

    class Model:

        def fit_score_predict(train, test, target_feature, normalize_target=False, random_state=0):
            X = train.drop(columns=[target_feature])
            y = train[target_feature]
            # Basic Scoring
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
            model = LinearRegression()
            model.fit(X_train, y_train)
            X_predictions = model.predict(X_test)
            score = math.sqrt(mean_squared_error(y_test, X_predictions))
            # Fit on full train data
            model.fit(X, y)
            predictions = model.predict(test)
            return (predictions, score)

        def save_predictions(test_df, predictions, target_feature, normalize_target):
            now = str(time.time()).split('.')[0]
            test_df[target_feature] = predictions
            if normalize_target:
                test_df[target_feature] = test_df[target_feature].apply(lambda x: np.expm1(x))
            test_df[[test_df.columns[0], target_feature]].to_csv('output/submit-'+ now + '.csv', index=False)


    explore = Explore()
    clean = Clean()
    engineer = Engineer()
    model = Model()

    DATA = {}
    DATA['BASE_PATH'] = '/'.join(__file__.split('/')[:-1])+'/'
    DATA['TRAIN'] = pd.read_csv(DATA['BASE_PATH'] + '../input/train.csv')
    DATA['TEST'] = pd.read_csv(DATA['BASE_PATH'] + '../input/test.csv')
    DATA['TARGET_FEATURE'] = 'SalePrice'
    DATA['IGNORE_FEATURES'] = ['Id']
    DATA['QUAL_FEATURES'] = explore.get_qual_features(DATA['TRAIN'], DATA['TARGET_FEATURE'], DATA['IGNORE_FEATURES'])
    DATA['QUANT_FEATURES'] = explore.get_quant_features(DATA['TRAIN'],DATA['TARGET_FEATURE'], DATA['IGNORE_FEATURES'])


    CONFIGS = pd.DataFrame([
        { # 0.119222 (LB: 0.12128)
            'sum': [],
            'multiply': [],
            'drop': [],
            'options': {
                'use_default_clean': True,
                'drop_corr': 0.1,
                'normalize_target': True,
                'normalize_quant_features': True,
                'skew_threshold': 0.4,
                'scale_encoded_qual_features': True,
                'scale_quant_features': True,
                'bath_porch_sf': True,
                'house_remodel_and_age': True
            }
        },
        # { 
        #     'sum': [],
        #     'multiply': [],
        #     'drop': [],
        #     'options': {
        #         'use_default_clean': False,
        #         'drop_corr': 0,
        #         'normalize_target': True,
        #         'normalize_quant_features': False,
        #         'skew_threshold': 0.0,
        #         'scale_encoded_qual_features': False,
        #         'scale_quant_features': False,
        #         'bath_porch_sf': True,
        #         'house_remodel_and_age': True
        #     }
        # },
        # { # 0.130841
        #     'sum': [
        #         ['GrLivArea', 'TotalBsmtSF'], 
        #         ['1stFlrSF', '2ndFlrSF']
        #     ],
        #     'multiply': [
        #         ['OverallQual', 'OverallCond'], 
        #         ['GarageQual_E', 'GarageCond_E'],
        #         ['ExterQual_E', 'ExterCond_E'],
        #         ['KitchenAbvGr', 'KitchenQual_E'],
        #         ['Fireplaces', 'FireplaceQu_E'],
        #         ['GarageArea', 'GarageQual_E_x_GarageCond_E'],
        #         ['PoolArea', 'PoolQC_E']
        #     ],
        #     'drop': ['BedroomAbvGr'],
        #     'options': {
        #         'use_default_clean': True,
        #         'drop_corr': 0.1,
        #         'normalize_target': True,
        #         'normalize_quant_features': True,
        #         'skew_threshold': 0.4,
        #         'scale_encoded_qual_features': True,
        #         'bath_porch_sf': True,
        #         'house_remodel_and_age': True
        #     },
        # },
    ])



    def score_configs(DATA, CONFIGS, times):
        scores_df = pd.DataFrame(columns=['configs', 'score', 'predictions', 'correlations', 'disparity'])
        default_qual_features_encoded, default_train_clean, default_test_clean = clean.run(DATA, CONFIGS.iloc[0])
        while times > 0:
            for index, CONFIG in CONFIGS.iterrows():
                if CONFIG['options']['use_default_clean'] is False:
                    qual_features_encoded, train_clean, test_clean = clean.run(DATA, CONFIG)
                else:
                    qual_features_encoded, train_clean, test_clean = (default_qual_features_encoded.copy(), default_train_clean.copy(), default_test_clean.copy())
                if CONFIG['options']['scale_quant_features']:
                    train_clean, test_clean = engineer.scale_quant_features(train_clean, test_clean, DATA['QUANT_FEATURES'])
                if CONFIG['options']['bath_porch_sf']:
                    train_clean, test_clean = engineer.bath_porch_sf(train_clean, test_clean)
                if CONFIG['options']['house_remodel_and_age']:
                    train_clean, test_clean = engineer.house_remodel_and_age(train_clean, test_clean)
                if len(CONFIG['sum']) > 0:
                    train_clean, test_clean = engineer.sum_features(train_clean, test_clean, CONFIG['sum'])
                if len(CONFIG['multiply']) > 0:
                    train_clean, test_clean = engineer.multiply_features(train_clean, test_clean, CONFIG['multiply'])
                correlations, disparity = explore.run(train_clean, qual_features_encoded, DATA['TARGET_FEATURE'])
                if CONFIG['options']['drop_corr'] > 0 or len(CONFIG['drop']) > 0:
                    train_clean, test_clean = engineer.drop_features(train_clean, test_clean, DATA['TARGET_FEATURE'], CONFIG['drop'], correlations=correlations, threshold=CONFIG['options']['drop_corr'])
                predictions, score = model.fit_score_predict(train_clean, test_clean, DATA['TARGET_FEATURE'], random_state=int(times ** 2))
                scores_df = scores_df.append({'configs': index, 'score': score, 'predictions': predictions, 'correlations': correlations, 'disparity': disparity}, ignore_index=True)
            times -= 1
        return scores_df

    scores_df = score_configs(DATA, CONFIGS, 10)

    save_config = 0
    model.save_predictions(DATA['TEST'], scores_df.iloc[save_config]['predictions'], DATA['TARGET_FEATURE'], CONFIGS.iloc[save_config]['options']['normalize_target'])

    print('######################')
    print(scores_df[['configs','score']].groupby('configs').mean())
    print('######################')
    print(scores_df[['configs','score']].sort_values('configs'))
    print('######################')
'''
