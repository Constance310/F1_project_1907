from data_preparation import baseline_data_prep, normal_data_prep
from data_cleaning import baseline_cleaning, normal_cleaning
from f1_project.params import *


if MODEL_VERSION == "light":
    # Data preparation
    df = baseline_data_prep()

    # Data cleaning
    df_baseline = baseline_cleaning(df)
    df_baseline.to_csv("raw_data/df_baseline2.csv", index=False)

if MODEL_VERSION == "normal":
    # Data preparation
    df = normal_data_prep()

    # Data cleaning
    df_baseline = normal_cleaning(df)
    df_baseline.to_csv("raw_data/df_baseline2.csv", index=False)
