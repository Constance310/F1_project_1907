import pandas as pd
import os
from f1_project.f1_packages.data_cleaning import *
from f1_project.f1_packages.data_preparation import *

def save_to_csv(df: pd.DataFrame, file_name):
    url = '../../raw_data'
    df.to_csv(f"{url}{file_name}.csv")


#def load_kaggle(): ## TO BE FINISHED
def get_original_driver(new_driver_number):
    df = get_data()
    dico = driver_dictionary(df)
    keys = [key for key, value in dico.items() if value[0] == new_driver_number]
    driver_code = dico[keys[0]][1]
    print(f"The driver's code name is {driver_code}")
