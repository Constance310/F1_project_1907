import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

def initialize_model(model, n_estimators=100, max_depth=4):
"""
Return the result depending on the chosen model
"""
if model=='LogReg':
    LogReg = LogisticRegression()
    return LogReg

if model=='randomforest':
    forest = RandomForestClassifier()
    return forest

if model=='XGBR':
    xgb_reg = XGBRegressor()
    return xgb_reg
