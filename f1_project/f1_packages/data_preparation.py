import kagglehub
import pandas as pd
import numpy as np
import os


def kaggle_to_df():
    """Getting the data from kaggle and turning it to dataframe with interesting columns"""

    # Downloading datasets
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pit_df = pd.read_csv(os.path.join(root_dir,"raw_data","pit_stops.csv"))
    lap_times_df = pd.read_csv(os.path.join(root_dir,"raw_data","lap_times.csv"))
    races_df = pd.read_csv(os.path.join(root_dir,"raw_data","races.csv"))

    # Rearanging pit dataset
    pit_df = pit_df[["raceId", "driverId", "stop", "lap", "time", "milliseconds"]].copy()
    pit_df.rename(columns={"stop": "#_cumul_stop"}, inplace=True)
    pit_df.rename(columns={"milliseconds": "pit_duration"}, inplace=True)

    # Rearanging races dataset
    races_df = races_df[["raceId", "name", "date"]]

    # Rearanging lap times dataset
    lap_times_df = lap_times_df[["raceId", "driverId", "lap", "position", "milliseconds"]]

    # Merging lap_times with races to get the dates and name of each race
    lap_times_df2 = pd.merge(lap_times_df, races_df, on="raceId")

    # Creating a cumul time column to have the commulative time of each pilot after each lap
    lap_times_df2["cumul_time"] = lap_times_df2.groupby(["raceId", "driverId"])["milliseconds"].cumsum()

    # Merging lap_times with pit datasets
    df1 = pd.merge(lap_times_df2, pit_df, how="left", on=["raceId", "driverId", "lap"])

    return df1


if __name__ == '__main__':
    kaggle_to_df()
