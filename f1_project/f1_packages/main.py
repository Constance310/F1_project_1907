from f1_project.f1_packages.data_preparation import *
from f1_project.f1_packages.data_cleaning import *
from f1_project.f1_packages.params import *
from f1_project.f1_packages.preprocessor import *
from f1_project.f1_packages.utils import *
import pandas as pd


def generation_dataframe(MODEL_VERSION="light"):
    if MODEL_VERSION == "light":
        # Data cleaning
        df_clean = light_cleaning()
        # Data preparation
        df_good = baseline_data_prep(df_clean)

        # df_baseline.to_csv("raw_data/df_baseline2.csv", index=False)

    if MODEL_VERSION == "normal":
        # Data cleaning
        df_clean = normal_cleaning()
        # Data preparation
        df_good = normal_data_prep(df_clean)

        # df_baseline.to_csv("raw_data/df_baseline2.csv", index=False)

    # save_to_csv(df_clean, "df_clean2")
    return df_good


def test_train_split(df):
    """Creating the X_train and X_test and downloading it into csv files"""
    # Separate X and y
    X = df.drop(columns=["undercut_tentative", "undercut_success"])
    y = df["undercut_success"]

    # Split data into train, test and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Saving the datasets
    save_to_csv(pd.DataFrame(X_train), "X_train")
    save_to_csv(pd.DataFrame(X_test), "X_test")

    # Returing X and y
    return X_train, X_test, y_train, y_test


# def preprocessing(X_train, X_test):
#     Xx

#     return X_train_preprocessed, X_test_preprocessed




# if __name__ == '__main__':
    # preprocessing(generation_dataframe(MODEL_VERSION))
