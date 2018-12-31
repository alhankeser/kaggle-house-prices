import numpy as np
import pandas as pd


def combine_features(dfs, feature_sets):
    modified_dfs = []
    for df in dfs:
        for feature_set in feature_sets:
            if len(feature_set) > 2:
                raise ValueError('Only put 2 vars at a time to combine.') 
            combined_name = '_'.join(feature_set[:])
            df[combined_name] = df[feature_set[0]] + df[feature_set[1]]
        df = df.drop(columns=feature_set)
        modified_dfs.append(df)
    return modified_dfs


def make_binary(dfs, target_feature):
    return True