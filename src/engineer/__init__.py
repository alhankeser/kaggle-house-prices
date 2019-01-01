import numpy as np
import pandas as pd


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

def combine_features(clean_train, clean_test, feature_sets):
    dfs = [clean_train, clean_test]
    result = pd.DataFrame([])
    for feature_set in feature_sets:
        if len(feature_set) > 2:
            raise ValueError('Only put 2 vars at a time to combine.') 
        combined_name = '_'.join(feature_set[:])
        dfs[0][combined_name] = dfs[0][feature_set[0]] + dfs[0][feature_set[1]]
        dfs[0] = dfs[0].drop(columns=feature_set)
        dfs[1][combined_name] = dfs[1][feature_set[0]] + dfs[1][feature_set[1]]
        dfs[1] = dfs[1].drop(columns=feature_set)
    return dfs


def make_binary(dfs, target_feature):
    return True