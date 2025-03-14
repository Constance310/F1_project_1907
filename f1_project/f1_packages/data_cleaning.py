import kagglehub
import pandas as pd
import numpy as np
import os


def light_remove_columns(df):
    """Removing the date and time columns for our baseline model"""

    df = df.drop(columns=["raceId", "date", "time"])
    return df


def remove_outliers(df):
    """Removing outliers in the data"""

    # Removing the laps with a time greater that 180,000 ms
    df1 = df[df["lap_time"] <= 180_000]
    # Removing the laps where the car stops for the 5th time or more
    df2 = df1[df1["cumul_stop"] <= 4]
    # Removing the laps where the pit stop is longer than 50,000 ms
    df3 = df2[df2["pit_duration"] <= 50_000]
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


def light_cleaning(df):
    df = light_remove_columns(df)
    df = remove_outliers(df)
    df = fill_na(df)
    df = rename_GP(df)
    return df


def normal_cleaning(df):
    df = remove_outliers(df)
    df = fill_na(df)
    df = rename_GP(df)
    # ADD OTHER FUNCTIONS
    return df


if __name__ == '__main__':
    light_remove_columns()
    remove_outliers()
    fill_na()
    rename_GP()
    light_cleaning()
    normal_cleaning()
