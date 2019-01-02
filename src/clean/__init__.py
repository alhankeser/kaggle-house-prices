import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import boxcox1p
import importlib


def run(DATA, CONFIG):
    train_clean = DATA['TRAIN'].copy()
    test_clean = DATA['TEST'].copy()
    normalize_target = CONFIG['options']['normalize_target']
    scale_encoded_qual_features = CONFIG['options']['scale_encoded_qual_features']
    target_feature = DATA['TARGET_FEATURE']
    quant_features = DATA['QUANT_FEATURES']
    normalize_quant_features = CONFIG['options']['normalize_quant_features']
    skew_threshold = CONFIG['options']['skew_threshold']

    # Create qual features encoding lookup table
    qual_features_encoded = create_encoding_lookup(train_clean.fillna(0), DATA['QUAL_FEATURES'], DATA['TARGET_FEATURE'])
    # Get skewed features based on training data
    skewed_features = get_skewed_features(train_clean, quant_features, skew_threshold)
    if scale_encoded_qual_features:
        qual_features_encoded = scale_qual_feature_encoding(qual_features_encoded, DATA['TARGET_FEATURE'])
    if normalize_target:
        train_clean[target_feature] = np.log1p(train_clean[target_feature])
    # Both
    dfs = [train_clean, test_clean]
    result = [qual_features_encoded]
    for df in dfs:
        df = df.fillna(0)
        df = encode_qual_features(df, qual_features_encoded)
        if normalize_quant_features:
            df = normalize_features(df, skewed_features)
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