import pandas as pd

def save_to_csv(X: pd.DataFrame):
    X.to_csv('../../raw_data/{X}.csv')
