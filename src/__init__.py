# External libraries
import pandas as pd
import importlib
import warnings
warnings.filterwarnings(action="ignore")


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
DATA['IGNORE_FEATURES'] = ['Id']
DATA['QUAL_FEATURES'] = explore.get_qual_features(DATA['TRAIN'])
DATA['QUANT_FEATURES'] = explore.get_quant_features(DATA['TRAIN'],DATA['TARGET_FEATURE'], DATA['IGNORE_FEATURES'])


# TODO: 
##  count encoded quals and remove ones with low sample size / pvalue
CONFIGS = pd.DataFrame([
     { # 0.120931
        'sum': [],
        'multiply': [],
        'drop': ['BedroomAbvGr'],
        'options': {
            'use_default_clean': True,
            'drop_corr': 0.1,
            'normalize_target': True,
            'normalize_quant_features': True,
            'skew_threshold': 0.4,
            'scale_encoded_qual_features': True,
            'bath_porch_sf': True,
            'house_remodel_and_age': True
        }
    },
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
    # qual_std = qual_features_encoded.groupby('encoded_name')['num_val'].std().sort_values()
    # print(qual_std)
    # print(disparity)
    print(train_clean.columns)
    return scores_df

scores_df = score_configs(DATA, CONFIGS, 10)

save_config = 0
model.save_predictions(DATA['TEST'], scores_df.iloc[save_config]['predictions'], DATA['TARGET_FEATURE'], CONFIGS.iloc[save_config]['options']['normalize_target'])

print('######################')
print(scores_df[['configs','score']].groupby('configs').mean())
print('######################')
print(scores_df[['configs','score']].sort_values('configs'))
print('######################')
