# External imports:
import numpy as np
import pandas as pd

# Local imports:
import explore
import clean
import engineer
import model

#
# DATA:
#
base_path = '/'.join(__file__.split('/')[:-1])
train_df = pd.read_csv(base_path + '/data/train.csv')
test_df = pd.read_csv(base_path + '/data/test.csv')
target_var = 'SalePrice'
ignore = ['Id', 'SalePrice']

#
# CLEAN:
#
train_clean = train_df.copy()
test_clean = test_df.copy()
dfs = [train_clean, test_clean]

train_clean = clean.transform_to_log(train_clean,target_var)


#
# EXPLORE:
#
explore.correlate(train_clean, target_var)
# explore.disparity(train_clean, target_var)