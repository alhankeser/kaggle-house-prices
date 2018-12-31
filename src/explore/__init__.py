import numpy as np
import pandas as pd
from scipy import stats

def correlate(df, target_var, method='spearman'):
    correlation_matrix = df.corr(method='spearman')
    correlation_matrix = correlation_matrix.sort_values(target_var)
    correlation_matrix = correlation_matrix.drop(target_var)
    return correlation_matrix[[target_var]]

# def disparity(df, target_var):
#     anova_df = pd.DataFrame()
#     anova_df['feature'] = df.drop(columns=target_var).columns
#     p_values = []
#     for col in df.drop(columns=target_var).columns:
#         samples = []
#         for unique_val in df[col].unique():
#             sample = df[df[col] == unique_val][target_var].values
#             samples.append(sample)
#         p_value = stats.f_oneway(*samples)[1]
#         p_values.append(p_value)
#     anova_df['p_value'] = p_values
#     anova_df['disparity'] = np.log(1./anova_df['p_value'].values)
#     return anova_df.sort_values('disparity')

def get_quant_vars(df):
    variables = [f for f in df.columns if df.dtypes[f] != 'object']
    return variables

def get_qual_vars(df):
    variables = [f for f in train_df.columns if train_df.dtypes[f] == 'object']
    return variables
