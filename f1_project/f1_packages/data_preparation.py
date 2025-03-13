import kagglehub
import pandas as pd
import numpy as np
import os


def kaggle_to_df():
    """Getting the data from kaggle and turning it to dataframe with interesting columns"""

    # Downloading datasets
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pit_df = pd.read_csv(os.path.join(root_dir,"raw_data","kaggle", "pit_stops.csv"))
    lap_times_df = pd.read_csv(os.path.join(root_dir,"raw_data", "kaggle", "lap_times.csv"))
    races_df = pd.read_csv(os.path.join(root_dir,"raw_data", "kaggle", "races.csv"))

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
    df = pd.merge(lap_times_df2, pit_df, how="left", on=["raceId", "driverId", "lap"])

    return df


def identify_rivals(df):
    """
    Identifies rival drivers for each driver based on raceId and lap.
    A 'rival' is defined as any driver within 5 seconds ahead of the current driver on the same lap.
    Rivalry detection starts only from lap 10.
    """
    # Initialize an empty list column to store rivals
    df["rivals"] = [[] for _ in range(len(df))]

    # Iterate through each race
    for race in df["raceId"].unique():
        race_data = df[df["raceId"] == race]  # Filter for each race

        # Iterate through each lap in the race
        for lap in race_data["lap"].unique():
            if lap < 10:  # Skip rivalry identification before lap 10
                continue

            lap_data = race_data[race_data["lap"] == lap]  # Filter for each lap

            # Iterate through the lap data and compare drivers
            for i in range(1, len(lap_data)):  # Start from 1 to compare with previous driver
                current_driver = lap_data.iloc[i]

                # Collect all rivals within 5 seconds
                for j in range(i):  # Check all drivers before the current one in the lap
                    previous_driver = lap_data.iloc[j]

                    # Check if the previous driver is ahead and within the 5 seconds range
                    time_diff = current_driver["cumul_time"] - previous_driver["cumul_time"]
                    if 0 < time_diff <= 5000:  # Within 5 seconds
                        df.loc[current_driver.name, "rivals"].append(int(previous_driver["driverId"]))

    df['rivals'] = df['rivals'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else x)
    return df


if __name__ == '__main__':
    kaggle_to_df()
    identify_rivals()
