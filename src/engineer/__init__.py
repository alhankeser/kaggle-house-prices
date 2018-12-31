import numpy as np
import pandas as pd


def drop_features(clean_train, clean_test, target_feature, correlations, threshold):
    dfs = [clean_train, clean_test]
    features_to_drop = correlations[(correlations[target_feature] <= threshold) & (correlations[target_feature] >= (threshold * -1))].index
    dfs[0].drop(columns=features_to_drop, inplace=True)
    dfs[1].drop(columns=features_to_drop, inplace=True)
    return dfs

def combine_features(dfs, feature_sets):
    result = pd.DataFrame([])
    for df in dfs:
        for feature_set in feature_sets:
            if len(feature_set) > 2:
                raise ValueError('Only put 2 vars at a time to combine.') 
            combined_name = '_'.join(feature_set[:])
            df[combined_name] = df[feature_set[0]] + df[feature_set[1]]
        df = df.drop(columns=feature_set)
        result.append(df)
    return result


def make_binary(dfs, target_feature):
    return True