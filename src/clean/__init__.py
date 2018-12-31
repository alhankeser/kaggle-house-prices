import numpy as np

def transform_to_log(df, target_var):
    df[target_var] = np.log1p(df[target_var])
    return df

