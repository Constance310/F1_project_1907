
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor

def initialize_model(model, n_estimators=100, learning_rate=0.1, max_depth=4):
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

    if model =='SGD':
        sgd = GradientBoostingClassifier()
        return sgd

def train_model(model, X, y) :
    """
    Fit the model and return a fitted_model
    """
    model = model.fit(X, y)
    return model

def evaluate(model, X, y):

    accuracy = model.score(X, y)
    print(f"L'accuracy : {accuracy}")
    print(f"Matrice confusion : {confusion_matrix(X ,y)}")



def pred (model, X, y) :
    """
    evaluate the choosen model
    """
    y_pred = model.predict(X, y)
    y_proba = model.predict_proba(X, y)
    print(f"Report : {classification_report}")

    return y_pred, y_proba
