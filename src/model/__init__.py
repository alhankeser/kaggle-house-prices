from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import math
import time


def fit_score_predict(train, test, target_feature, normalize_target=False, random_state=0):
    X = train.drop(columns=[target_feature])
    y = train[target_feature]
    # Basic Scoring
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    model = LinearRegression()
    model.fit(X_train, y_train)
    X_predictions = model.predict(X_test)
    score = math.sqrt(mean_squared_error(y_test, X_predictions))
    # Fit on full train data
    model.fit(X, y)
    predictions = model.predict(test)
    return (predictions, score)

def save_predictions(test_df, predictions, target_feature, normalize_target):
    now = str(time.time()).split('.')[0]
    test_df[target_feature] = predictions
    if normalize_target:
        test_df[target_feature] = test_df[target_feature].apply(lambda x: np.expm1(x))
    test_df[[test_df.columns[0], target_feature]].to_csv('output/submit-'+ now + '.csv', index=False)