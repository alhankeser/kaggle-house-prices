import numpy as np
import pandas as pd
import operator
import functools

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