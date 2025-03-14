from data_preparation import baseline_data_prep, normal_data_prep
from data_cleaning import light_cleaning, normal_cleaning
from f1_project.f1_packages.params import *


def generation_dataframe(MODEL_VERSION):
    if MODEL_VERSION == "light":
        # Data preparation
        df = baseline_data_prep()

        # Data cleaning
        df_baseline = light_cleaning(df)
        df_baseline.to_csv("raw_data/df_baseline2.csv", index=False)

    if MODEL_VERSION == "normal":
        # Data preparation
        df = normal_data_prep()

        # Data cleaning
        df_baseline = normal_cleaning(df)
        # df_baseline.to_csv("raw_data/df_baseline2.csv", index=False)

    return df
