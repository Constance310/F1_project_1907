
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

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

def pred (model, X, y) :
    """
    evaluate the choosen model
    """
    y_pred = model.predict(X)
    #y_proba = model.predict_proba(X, y)
    report = classification_report(y, y_pred)
    print(f"Report : \n{report}")
    cm = confusion_matrix(y, y_pred)
    # Utiliser seaborn pour l'affichage
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])

    # Ajouter des titres et des labels
    plt.title('Matrice de Confusion')
    plt.xlabel('Prédictions')
    plt.ylabel('Véritables valeurs')
    plt.show()

    return y_pred
