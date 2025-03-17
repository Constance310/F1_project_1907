import kagglehub
import pandas as pd
import numpy as np
import os
from f1_project.f1_packages.params import *


def get_data():
    """Getting the data from kaggle and turning it to dataframe with interesting columns"""

    # Downloading datasets
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pit_df = pd.read_csv(os.path.join(root_dir,"raw_data","kaggle", "pit_stops.csv"))
    lap_times_df = pd.read_csv(os.path.join(root_dir,"raw_data", "kaggle", "lap_times.csv"))
    races_df = pd.read_csv(os.path.join(root_dir,"raw_data", "kaggle", "races.csv"))

    # Rearanging pit dataset and renaming
    pit_df = pit_df[["raceId", "driverId", "stop", "lap", "time", "milliseconds"]].copy()
    pit_df.rename(columns={"stop": "cumul_stop"}, inplace=True)
    pit_df.rename(columns={"milliseconds": "pit_duration"}, inplace=True)

    # Rearanging races dataset
    races_df = races_df[["raceId", "name", "date"]]

    # Rearanging lap times dataset and renaming
    lap_times_df = lap_times_df[["raceId", "driverId", "lap", "position", "milliseconds"]]
    lap_times_df.rename(columns={"milliseconds": "lap_time"}, inplace=True)

    # Merging lap_times with races to get the dates and name of each race
    lap_times_df2 = pd.merge(lap_times_df, races_df, on="raceId")

    # Creating a cumul time column to have the commulative time of each pilot after each lap
    lap_times_df2["cumul_time"] = lap_times_df2.groupby(["raceId", "driverId"])["lap_time"].cumsum()

    # Merging lap_times with pit datasets
    df = pd.merge(lap_times_df2, pit_df, how="left", on=["raceId", "driverId", "lap"])

    # Removing data prior to 2010
    df["date"] = pd.to_datetime(df["date"])
    df2 = df[df["date"].dt.year >= 2010]

    # Sort the values
    df3 = df2.sort_values(by=["raceId", "lap", "cumul_time"]).copy()
    df3 = df3.reset_index(drop=True)

    return df3


def remove_outliers(df):
    """Removing outliers in the data"""

    # Removing the laps with a time greater that 180,000 ms
    df1 = df[df["lap_time"] <= LAP_DURATION_MAX]
    # Removing the laps where the car stops for the 5th time or more
    df2 = df1[df1["cumul_stop"] <= NUMBER_PIT_MAX]
    # Removing the laps where the pit stop is longer than 50,000 ms
    df3 = df2[df2["pit_duration"] <= PIT_DURATION_MAX]
    return df3


def fill_na(df):
    """Changing the NaN values to 0"""

    df["cumul_stop"].fillna(method='bfill', inplace=True)
    df["cumul_stop"].fillna(0, inplace=True)
    df["pit_duration"].fillna(0, inplace=True)
    return df


def rename_GP(df):
    """Renaming the some GP whose names changed during years"""

    df['name'] = df['name'].replace({'70th Anniversary Grand Prix': 'British Grand Prix', 'Mexican Grand Prix': 'Mexico City Grand Prix'})
    return df


###################### PACKAGING ALL THE FUNCTIONS ##############################

def light_cleaning():
    df = get_data()
    df = remove_outliers(df)
    df = fill_na(df)
    df = rename_GP(df)
    return df


def normal_cleaning():
    df = get_data()
    df = remove_outliers(df)
    df = fill_na(df)
    df = rename_GP(df)
    # ADD OTHER FUNCTIONS
    return df


if __name__ == '__main__':
    get_data()
    remove_outliers()
    fill_na()
    rename_GP()
    light_cleaning()
    normal_cleaning()
