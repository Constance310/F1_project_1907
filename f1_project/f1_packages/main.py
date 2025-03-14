from f1_project.f1_packages.data_preparation import baseline_data_prep, normal_data_prep
from f1_project.f1_packages.data_cleaning import light_cleaning, normal_cleaning
from f1_project.f1_packages.params import *
from sklearn.model_selection import train_test_split
from f1_project.f1_packages.utils import save_to_csv
import pandas as pd

def generation_dataframe(MODEL_VERSION="light"):
    if MODEL_VERSION == "light":
        # Data preparation
        df = baseline_data_prep()
        # Data cleaning
        df_clean = light_cleaning(df)
        # df_baseline.to_csv("raw_data/df_baseline2.csv", index=False)
    if MODEL_VERSION == "normal":
        # Data preparation
        df = normal_data_prep()
        # Data cleaning
        df_clean = normal_cleaning(df)
        # df_baseline.to_csv("raw_data/df_baseline2.csv", index=False)
    return df_clean
