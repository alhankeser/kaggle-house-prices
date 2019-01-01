import numpy as np
import pandas as pd
import importlib


def run(DATA, CONFIG):
    train_clean = DATA['TRAIN'].copy()
    test_clean = DATA['TEST'].copy()
    normalize_target = CONFIG['options']['normalize_target']
    normalize_after_encode= CONFIG['options']['normalize_after_encode']
    scale_encoded_qual_features = CONFIG['options']['scale_encoded_qual_features']
    scale_threshold= CONFIG['options']['scale_threshold']
    target_feature = DATA['TARGET_FEATURE']


    # Train
    if normalize_target and (normalize_after_encode == False):
        train_clean[target_feature] = np.log1p(train_clean[target_feature])
    
    # Create qual features encoding lookup table
    qual_features_encoded = create_encoding_lookup(train_clean.fillna(0), DATA['QUAL_FEATURES'], DATA['TARGET_FEATURE'])
    
    if scale_encoded_qual_features:
        qual_features_encoded = scale_qual_feature_encoding(qual_features_encoded, DATA['TARGET_FEATURE'], scale_threshold)
   
    if normalize_target and (normalize_after_encode == True):
        train_clean[target_feature] = np.log1p(train_clean[target_feature])

    # Both
    dfs = [train_clean, test_clean]
    result = [qual_features_encoded]
    for df in dfs:
        df = df.fillna(0)
        df = encode_qual_features(df, qual_features_encoded)
        df = df.fillna(0)
        df = df.drop(columns=DATA['IGNORE_FEATURES'])
        result.append(df)
    return result


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

def scale_qual_feature_encoding(qual_features_encoded, target_feature, scale_threshold=0):
    result = qual_features_encoded.copy()
    for feature in result['feature'].unique():
        values = result[result['feature'] == feature]['num_val'].values
        medians = result[result['feature'] == feature][target_feature + '_median'].values
        for median in medians:
            scaled_value = 0
            if len(values) > scale_threshold:
                # scaled_value_max = len(values) * (median / medians.max())
                scaled_value = ((values.min() + 1) * (median / medians.min()))-1
            result.loc[(result['feature'] == feature) & (result[target_feature + '_median'] == median), 'num_val'] = scaled_value
    return result