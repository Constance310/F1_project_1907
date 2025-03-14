from sklearn.metrics import confusion_matrix, classification_report



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
