import numpy as np
import pandas as pd
from scipy import stats

def run(train_clean, qual_features_encoded, target_feature):
    qual_features_list = qual_features_encoded['feature'].unique()
    correlations = get_correlations(train_clean, target_feature)
    disparity = False # get_qual_feature_disparity(train_clean, qual_features_list, target_feature)
    # effect_size = compare_qual_feature_value_effect(train_clean, qual_features_list, target_feature)
    return (correlations, disparity)

def get_quant_features(df, target_feature, ignore_features):
    ignore_features = ignore_features.copy()
    features = [f for f in df.columns if df.dtypes[f] != 'object']
    ignore_features.append(target_feature)
    features = list(set(features) - set(ignore_features))
    return features


def get_qual_features(df):
    features = [f for f in df.columns if df.dtypes[f] == 'object']
    return features


def get_correlations(df, target_feature, method='spearman'):
    correlation_matrix = df.corr(method=method)
    correlation_matrix = correlation_matrix.sort_values(target_feature)
    correlation_matrix = correlation_matrix.drop(target_feature)
    return correlation_matrix[[target_feature]]


def get_qual_feature_disparity(df, qual_features_list, target_feature):
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

def compare_qual_feature_value_effect(df, qual_features_list, target_feature):
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
