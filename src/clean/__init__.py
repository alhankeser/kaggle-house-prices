import numpy as np
import pandas as pd


def run(DATA, CONFIG):
    train_clean = DATA['TRAIN'].copy()
    test_clean = DATA['TEST'].copy()
    
    # Train
    train_clean = transform_to_log(train_clean, DATA['TARGET_FEATURE'])
    train_clean, qual_features_encoded = encode_qual_features_train(train_clean, DATA['QUAL_FEATURES'], DATA['TARGET_FEATURE'])
    
    # Test
    test_clean = encode_qual_features_test(test_clean, qual_features_encoded)
    
    # Both
    dfs = [train_clean, test_clean]
    result = [qual_features_encoded['encoded_name'].unique()]
    for df in dfs:
        df = df.fillna(0)
        df = df.drop(columns=DATA['IGNORE_FEATURES'])
        result.append(df)
    return result


def fillna(df, fill_with=0):
    return df.fillna(fill_with)


def transform_to_log(df, target_feature):
    df[target_feature] = np.log1p(df[target_feature])
    return df


def untransform_from_log(df, target_feature):
    df[target_feature] = df[target_feature].apply(lambda x: np.expm1(x))
    return df


def encode_qual_features_train(df, qual_features, target_feature, suffix='_E'):
    qual_features_encoded = pd.DataFrame(columns=['feature_name','encoded_name','value','order'])
    encoded_df = df.copy()
    for qual_feature in qual_features:
        qual_feature_encoded_name = qual_feature + suffix
        order_df = pd.DataFrame()
        order_df['value'] = df[qual_feature].unique() 
        order_df.index = order_df.value
        order_df[target_feature + '_median'] = df[[qual_feature, target_feature]].groupby(qual_feature)[[target_feature]].median()
        order_df = order_df.sort_values(target_feature + '_median')
        order_df['order'] = range(1, len(order_df)+1)
        order_df = order_df['order'].to_dict()
        for qual_val, order in order_df.items():
            encoded_df.loc[encoded_df[qual_feature] == qual_val, qual_feature_encoded_name] = order
            qual_features_encoded = qual_features_encoded.append({'feature_name': qual_feature, 'encoded_name': qual_feature_encoded_name, 'value': qual_val, 'order': order}, ignore_index=True)
    encoded_df = encoded_df.drop(columns=qual_features)
    return (encoded_df, qual_features_encoded)


def encode_qual_features_test(df, qual_features_encoded):
    encoded_df = df.copy()
    for encoded_index, encoded_row in qual_features_encoded.iterrows():
        feature = encoded_row['feature_name']
        encoded_name = encoded_row['encoded_name']
        value = encoded_row['value']
        order = encoded_row['order'] 
        encoded_df.loc[encoded_df[feature] == value, encoded_name] = order
    encoded_df = encoded_df.drop(columns=qual_features_encoded['feature_name'].unique())
    return encoded_df