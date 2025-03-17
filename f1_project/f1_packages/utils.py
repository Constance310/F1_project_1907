import pandas as pd
import os

def save_to_csv(df: pd.DataFrame, file_name):
    df.to_csv(f"raw_data/{file_name}.csv")


#def load_kaggle(): ## TO BE FINISHED
