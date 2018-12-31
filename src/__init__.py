# External libraries
import pandas as pd
import importlib


# Local modules
explore = importlib.import_module('explore')
clean = importlib.import_module('clean')
engineer = importlib.import_module('engineer')
model = importlib.import_module('model')


# DATA
DATA = {}
DATA['BASE_PATH'] = '/'.join(__file__.split('/')[:-1])+'/'
DATA['TRAIN'] = pd.read_csv(DATA['BASE_PATH'] + 'data/train.csv')
DATA['TEST'] = pd.read_csv(DATA['BASE_PATH'] + 'data/test.csv')
DATA['TARGET_FEATURE'] = 'SalePrice'
DATA['QUAL_FEATURES'] = explore.get_qual_features(DATA['TRAIN'])
DATA['QUANT_FEATURES'] = explore.get_quant_features(DATA['TRAIN'])
DATA['IGNORE_FEATURES'] = ['Id']

CONFIGS = pd.DataFrame([
    # WORK IN PROGRESS...
    # {
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0,
    #         'normalize_target': False,
    #         'scale_encoded_qual_features': False
    #     }
    # },
    # {
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0,
    #         'normalize_target': True,
    #         'scale_encoded_qual_features': False
    #     }
    # },
    # {
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0,
    #         'normalize_target': True,
    #         'scale_encoded_qual_features': True
    #     }
    # },
    {
        'combine': [],
        'drop': [],
        'options': {
            'drop_corr': 0.1,
            'normalize_target': True,
            'scale_encoded_qual_features': True
        }
    },
    # {
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0.6,
    #         'normalize_target': True,
    #         'scale_encoded_qual_features': True
    #     }
    # }

])

def score_configs(DATA, CONFIGS, times):
    scores_df = pd.DataFrame(columns=['configs', 'score', 'predictions', 'correlations', 'disparity'])
    while times > 0:
        for index, config in CONFIGS.iterrows():
            qual_features_encoded, train_clean, test_clean = clean.run(DATA, config)
            correlations, disparity = explore.run(train_clean, qual_features_encoded, DATA['TARGET_FEATURE'])
            if config['options']['drop_corr'] > 0:
                train_clean, test_clean = engineer.drop_features(train_clean, test_clean, DATA['TARGET_FEATURE'], correlations, config['options']['drop_corr'])
            predictions, score = model.fit_score_predict(train_clean, test_clean, DATA['TARGET_FEATURE'])
            scores_df = scores_df.append({'configs': index, 'score': score, 'predictions': predictions, 'correlations': correlations, 'disparity': disparity}, ignore_index=True)
        times -= 1
    return scores_df

scores_df = score_configs(DATA, CONFIGS, 1)

save_config = 0
model.save_predictions(DATA['TEST'], scores_df.iloc[save_config]['predictions'], DATA['TARGET_FEATURE'], CONFIGS.iloc[save_config]['options']['normalize_target'])

# print(scores_df[['configs','score']].groupby('configs').mean())

print(scores_df.iloc[save_config]['predictions'])

# print(scores_df.iloc[1]['correlations'])
# correlations = pd.DataFrame(scores_df.iloc[1]['correlations'])

# target_feature = 'SalePrice'
# threshold = 0.2
# print(correlations[(correlations[target_feature] <= threshold) & (correlations[target_feature] >= (threshold * -1))].index)
# print(correlations[(correlations[target_feature] <= threshold) & (correlations[target_feature] >= threshold*-1)].index)
