import pandas as pd
import os

def save_to_csv(df: pd.DataFrame, file_name):
    url = '../../raw_data/'
    df.to_csv(f"{url}{file_name}.csv")


#def load_kaggle(): ## TO BE FINISHED
