import kagglehub
import pandas as pd
import numpy as np
import os
from f1_project.f1_packages.params import *


def get_data():
    """Getting the data from kaggle and turning it to dataframe with interesting columns"""
    print("\n🔄 Loading data from CSV files...")

    # Downloading datasets
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pit_df = pd.read_csv(os.path.join(root_dir,"raw_data","kaggle", "pit_stops.csv"))
    lap_times_df = pd.read_csv(os.path.join(root_dir,"raw_data","kaggle", "lap_times.csv"))
    races_df = pd.read_csv(os.path.join(root_dir,"raw_data","kaggle", "races.csv"))
    print(f"✅ Loaded initial datasets - Shapes:")
    print(f"   • Pit stops: {pit_df.shape}")
    print(f"   • Lap times: {lap_times_df.shape}")
    print(f"   • Races: {races_df.shape}")

    # Rearanging pit dataset and renaming
    pit_df = pit_df[["raceId", "driverId", "stop", "lap", "time", "milliseconds"]].copy()
    pit_df.rename(columns={"stop": "cumul_stop"}, inplace=True)
    pit_df.rename(columns={"milliseconds": "pit_duration"}, inplace=True)
    print("\n✅ Pit stops data reorganized")

    # Rearanging races dataset
    races_df = races_df[["raceId", "name", "date"]]
    print("✅ Races data reorganized")

    # Rearanging lap times dataset and renaming
    lap_times_df = lap_times_df[["raceId", "driverId", "lap", "position", "milliseconds"]]
    lap_times_df.rename(columns={"milliseconds": "lap_time"}, inplace=True)
    print("✅ Lap times data reorganized")

    # Merging lap_times with races to get the dates and name of each race
    lap_times_df2 = pd.merge(lap_times_df, races_df, on="raceId")
    print(f"✅ Merged lap times with races data: {lap_times_df2.shape}")

    # Creating a cumul time column
    lap_times_df2["cumul_time"] = lap_times_df2.groupby(["raceId", "driverId"])["lap_time"].cumsum()
    print("✅ Added cumulative time column")

    # Merging lap_times with pit datasets
    df = pd.merge(lap_times_df2, pit_df, how="left", on=["raceId", "driverId", "lap"])
    print(f"✅ Merged with pit stops data: {df.shape}")

    # Removing data prior to 2010
    df["date"] = pd.to_datetime(df["date"])
    df2 = df[df["date"].dt.year >= 2010]
    print(f"✅ Filtered data from 2010 onwards: {df2.shape}")

    # Sort the values
    df3 = df2.sort_values(by=["raceId", "lap", "cumul_time"]).copy()
    df3 = df3.reset_index(drop=True)
    print("✅ Data sorted and index reset")

    return df3

def fill_na(df):
    """Changing the NaN values to 0"""
    print("\n📊 Filling missing values...")
    initial_na = df.isna().sum()
    print(f"Initial NA counts:\n{initial_na[initial_na > 0]}")

    df["cumul_stop"].fillna(method='bfill', inplace=True)
    df["cumul_stop"].fillna(0, inplace=True)
    df["pit_duration"].fillna(0, inplace=True)

    final_na = df.isna().sum()
    print(f"\nFinal NA counts:\n{final_na[final_na > 0]}")
    print("✅ Missing values handled")
    return df

def remove_outliers(df):
    """Removing outliers in the data"""
    print("\n🔍 Removing outliers...")
    initial_shape = df.shape

    # Removing the laps with a time greater that 180,000 ms
    df1 = df[df["lap_time"] <= int(LAP_DURATION_MAX)]
    print(f"✅ Removed lap times > {LAP_DURATION_MAX}ms: {initial_shape[0] - df1.shape[0]} rows")

    # Removing the laps where the car stops for the 5th time or more
    df2 = df1[df1["cumul_stop"] <= int(NUMBER_PIT_MAX)]
    print(f"✅ Removed pit stops > {NUMBER_PIT_MAX}: {df1.shape[0] - df2.shape[0]} rows")

    # Removing the laps where the pit stop is longer than 50,000 ms
    df3 = df2[df2["pit_duration"] <= int(PIT_DURATION_MAX)]
    print(f"✅ Removed pit durations > {PIT_DURATION_MAX}ms: {df2.shape[0] - df3.shape[0]} rows")

    print(f"\nTotal rows removed: {initial_shape[0] - df3.shape[0]}")
    print(f"Final shape: {df3.shape}")
    return df3

def rename_GP(df):
    """Renaming the some GP whose names changed during years"""
    print("\n🏁 Standardizing Grand Prix names...")
    initial_unique = df['name'].nunique()
    print(f"Initial unique GP names: {initial_unique}")

    df['name'] = df['name'].replace({
        '70th Anniversary Grand Prix': 'British Grand Prix',
        'Mexican Grand Prix': 'Mexico City Grand Prix'
    })

    final_unique = df['name'].nunique()
    print(f"Final unique GP names: {final_unique}")
    print(f"✅ Standardized {initial_unique - final_unique} GP names")
    return df


def light_cleaning():
    """Light cleaning of the data"""
    print("\n🧹 Starting light cleaning process...")

    print("\n1️⃣ Getting data...")
    df = get_data()

    print("\n2️⃣ Filling missing values...")
    df = fill_na(df)

    print("\n3️⃣ Removing outliers...")
    df = remove_outliers(df)

    print("\n4️⃣ Standardizing GP names...")
    df = rename_GP(df)

    print("\n✨ Light cleaning completed!")
    print(f"Final dataset shape: {df.shape}")
    return df

def normal_cleaning():
    """Normal cleaning of the data with additional steps"""
    print("\n🧼 Starting normal cleaning process...")

    print("\n1️⃣ Getting data...")
    df = get_data()

    print("\n3️⃣ Filling missing values...")
    df = fill_na(df)

    print("\n2️⃣ Removing outliers...")
    df = remove_outliers(df)

    print("\n4️⃣ Standardizing GP names...")
    df = rename_GP(df)
    # ADD OTHER FUNCTIONS

    print("\n✨ Normal cleaning completed!")
    print(f"Final dataset shape: {df.shape}")
    return df

if __name__ == '__main__':
    print("\n🚀 Starting data cleaning process...")
    get_data()
    fill_na()
    remove_outliers()
    rename_GP()
    light_cleaning()
    normal_cleaning()
    print("\n✅ All cleaning processes completed!")
