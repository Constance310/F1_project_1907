from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def initialize_model(
    model,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    C=1.0,
    solver='lbfgs',
    max_iter=1000
):
    """
    Initialize and return a model with specified parameters.

    Args:
        model (str): Type of model to initialize ('LogReg', 'randomforest', 'XGBC', 'SGD')
        n_estimators (int): Number of trees/estimators for ensemble methods
        learning_rate (float): Learning rate for gradient-based methods
        max_depth (int): Maximum depth of trees
        random_state (int): Random seed for reproducibility
        min_samples_split (int): Minimum samples required to split a node
        min_samples_leaf (int): Minimum samples required at a leaf node
        max_features (str): Number of features to consider when looking for the best split
        C (float): Inverse of regularization strength for LogisticRegression
        solver (str): Algorithm to use in LogisticRegression
        max_iter (int): Maximum number of iterations for solvers

    Returns:
        Initialized model with specified parameters
    """
    if model == 'LogReg':
        return LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=-1
        )

    if model == 'randomforest':
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1
        )

    if model == 'XGBC':
        return XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )

    if model == 'SGD':
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

    raise ValueError(f"Model type '{model}' not recognized. Choose from: 'LogReg', 'randomforest', 'XGBC', 'SGD'")

def train_model(model, X, y):
    """
    Fit the model and return a fitted_model
    """
    model = model.fit(X, y)
    return model

def pred(model, X, y):
    """
    evaluate the chosen model
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)  # Added probability predictions

    # Print classification report
    report = classification_report(y, y_pred)
    print(f"Report : \n{report}")

    # Create confusion matrix visualization
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Undercut", "Undercut"],
                yticklabels=["No Undercut", "Undercut"])

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual values')
    plt.show()

    return y_pred, y_proba
