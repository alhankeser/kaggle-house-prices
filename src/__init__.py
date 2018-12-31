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


CONFIG = [
    # WORK IN PROGRESS...
    {
        'combine': [['GrLivArea', 'TotalBsmtSF']],
        'drop': [],
        'options': {
            'normalize_target': True
        }
    }
]

# CLEAN:
qual_features_encoded, train_clean, test_clean = clean.run(DATA, CONFIG)

# ENGINEER:
train_clean, test_clean = engineer.combine_features([train_clean, test_clean], [['GrLivArea', 'TotalBsmtSF']])

# EXPLORE:
correlations, disparity, effect_size = explore.run(train_clean, qual_features_encoded, DATA['TARGET_FEATURE'])

print(correlations.head(5))
print(disparity.head(5))
print(effect_size.head(10))