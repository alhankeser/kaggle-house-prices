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
    # { # 0.154847
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0,
    #         'normalize_target': True,
    #         'scale_encoded_qual_features': False
    #     }
    # },
    # {  # 0.157314
    #     'combine': [['TotalBsmtSF','GrLivArea']],
    #     'drop': ['BsmtFinSF1', 'BsmtFinSF2', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF'],
    #     'options': {
    #         'drop_corr': 0,
    #         'normalize_target': True,
    #         'scale_encoded_qual_features': False
    #     }
    # },
    # {  # 0.154901
    #     'combine': [['TotalBsmtSF','GrLivArea']],
    #     'drop': ['MoSold', 'YrSold'],
    #     'options': {
    #         'drop_corr': 0,
    #         'normalize_target': True,
    #         'scale_encoded_qual_features': False
    #     }
    # },
    # {  # 0.155069
    #     'combine': [],
    #     'drop': ['MoSold', 'YrSold', 'BsmtFinSF1', 'BsmtFinSF2', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF'],
    #     'options': {
    #         'drop_corr': 0,
    #         'normalize_target': True,
    #         'scale_encoded_qual_features': False
    #     }
    # },
    # {  # 0.157415
    #     'combine': [['TotalBsmtSF','GrLivArea']],
    #     'drop': ['MoSold', 'YrSold', 'BsmtFinSF1', 'BsmtFinSF2', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF'],
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
    #         'normalize_after_encode': False,
    #         'scale_encoded_qual_features': True,
    #         'scale_threshold': -1
    #     }
    # },
    # { 
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0,
    #         'normalize_target': True,
    #         'normalize_after_encode': True,
    #         'scale_encoded_qual_features': True,
    #         'scale_threshold': -1
    #     }
    # },
    # { # 0.153464
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0,
    #         'normalize_target': True,
    #         'normalize_after_encode': True,
    #         'scale_encoded_qual_features': True,
    #         'scale_threshold': 10
    #     }
    # },
    # { # 0.15163
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0,
    #         'normalize_target': True,
    #         'normalize_after_encode': True,
    #         'scale_encoded_qual_features': True,
    #         'scale_threshold': 10
    #     }
    # },
    # { # 0.151401
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0,
    #         'normalize_target': True,
    #         'normalize_after_encode': True,
    #         'scale_encoded_qual_features': True,
    #         'scale_threshold': 20
    #     }
    # },
    #  { # 0.14438 (LB: 0.12951)
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0.1,
    #         'normalize_target': True,
    #         'normalize_quant_features': False,
    #         'scale_encoded_qual_features': False
    #     }
    # },
    #   { # 0.132941 (LB: 0.12499)
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0.1,
    #         'normalize_target': True,
    #         'normalize_quant_features': True,
    #         'scale_encoded_qual_features': False
    #     }
    # },
    #   { #0.131128 (LB: 0.12411)
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0.15,
    #         'normalize_target': True,
    #         'normalize_quant_features': True,
    #         'scale_encoded_qual_features': True
    #     }
    # },
     # { # 0.150094
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0.1,
    #         'normalize_target': True,
    #         'normalize_after_encode': True,
    #         'scale_encoded_qual_features': True,
    #         'scale_threshold': 10
    #     }
    # },
    # { #0.149946
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0.1,
    #         'normalize_target': True,
    #         'normalize_after_encode': True,
    #         'scale_encoded_qual_features': True,
    #         'scale_threshold': 15
    #     }
    # }
    #   { # 0.131195
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0.1,
    #         'normalize_target': True,
    #         'normalize_quant_features': True,
    #         'scale_encoded_qual_features': True
    #     }
    # },
    # { # on skew > 0.75 boxcox 0.130584 / log: 0.130318
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0.1,
    #         'normalize_target': True,
    #         'normalize_quant_features': True,
    #         'scale_encoded_qual_features': True,
    #         'skew_threshold': 0.75
    #     }
    # }
    # { 
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0.1,
    #         'normalize_target': True,
    #         'normalize_quant_features': True,
    #         'scale_encoded_qual_features': True,
    #         'skew_threshold': 0
    #     }
    # },
    # { 
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0.1,
    #         'normalize_target': True,
    #         'normalize_quant_features': True,
    #         'scale_encoded_qual_features': True,
    #         'skew_threshold': 0.1
    #     }
    # },
    # { 
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0.1,
    #         'normalize_target': True,
    #         'normalize_quant_features': True,
    #         'scale_encoded_qual_features': True,
    #         'skew_threshold': 0.2
    #     }
    # },
    # { 
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0.1,
    #         'normalize_target': True,
    #         'normalize_quant_features': True,
    #         'scale_encoded_qual_features': True,
    #         'skew_threshold': 0.3
    #     }
    # },
     {
        'sum': [],
        'multiply': [['KitchenQual', 'KitchenAbvGr']],
        'drop': [],
        'options': {
            'drop_corr': 0.1,
            'normalize_target': True,
            'normalize_quant_features': True,
            'skew_threshold': 0.4,
            'scale_encoded_qual_features': True
        }
    },
    #  { 
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0.1,
    #         'normalize_target': True,
    #         'normalize_quant_features': True,
    #         'scale_encoded_qual_features': True,
    #         'skew_threshold': 0.5
    #     }
    # },
    #  { 
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0.1,
    #         'normalize_target': True,
    #         'normalize_quant_features': True,
    #         'scale_encoded_qual_features': True,
    #         'skew_threshold': 0.6
    #     }
    # },
    #    { # 0.130120
    #     'combine': [],
    #     'drop': [],
    #     'options': {
    #         'drop_corr': 0.1,
    #         'normalize_target': True,
    #         'normalize_quant_features': True,
    #         'scale_encoded_qual_features': True,
    #         'skew_threshold': 0.7
    #     }
    # }
])

def score_configs(DATA, CONFIGS, times):
    scores_df = pd.DataFrame(columns=['configs', 'score', 'predictions', 'correlations', 'disparity'])
    while times > 0:
        for index, CONFIG in CONFIGS.iterrows():
            qual_features_encoded, train_clean, test_clean = clean.run(DATA, CONFIG)
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
    # print(train_clean.head())
    return scores_df

scores_df = score_configs(DATA, CONFIGS, 1)

save_config = 0
model.save_predictions(DATA['TEST'], scores_df.iloc[save_config]['predictions'], DATA['TARGET_FEATURE'], CONFIGS.iloc[save_config]['options']['normalize_target'])

print('######################')
print(scores_df[['configs','score']].groupby('configs').mean())
print('######################')
print(scores_df[['configs','score']].sort_values('configs'))
print('######################')
