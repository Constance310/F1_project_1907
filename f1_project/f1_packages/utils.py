import pandas as pd
import os
from f1_project.f1_packages.data_preparation import *

def save_to_csv(df: pd.DataFrame, file_name):
    url = '../../raw_data/'
    df.to_csv(f"{url}{file_name}.csv")


#def load_kaggle(): ## TO BE FINISHED
def get_original_driver(new_driver_number):
    df = get_data()
    dico = driver_dictionary(df)
    keys = [key for key, value in dico.items() if value[0] == new_driver_number]
    driver_code = dico[keys[0]][1]
    print(f"The driver's code name is {driver_code}")

def add_data_laps_fastf1():
    """Getting the data from fastf1 for laps and merge it to our dataset"""
    # GET DATA
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    df2018 = pd.read_csv(os.path.join(root_dir,"raw_data","fastf1", "fastf1_laps_2018_2021.csv"))
    df2022 = pd.read_csv(os.path.join(root_dir,"raw_data","fastf1", "fastf1_laps_2022.csv"))
    df2023 = pd.read_csv(os.path.join(root_dir,"raw_data","fastf1", "fastf1_laps_2023.csv"))
    df2024 = pd.read_csv(os.path.join(root_dir,"raw_data","fastf1", "fastf1_laps_2024.csv"))
    fast_f1 = pd.concat([df2018, df2022, df2023, df2024], axis=0) # On concatène les données

    # Rearranging dataset
    fast_f1 = fast_f1[['Driver', 'DriverNumber', 'LapNumber', 'Stint', 'Compound', 'TyreLife', 'FreshTyre', 'LapStartDate']]
    fast_f1["Name_numb"] = fast_f1["Driver"].astype(str) + "_" + fast_f1["DriverNumber"].astype(str)
    fast_f1['LapStartDate'] = fast_f1['LapStartDate'].str.slice(0, 10)
    fast_f1['Name_numb'] = fast_f1['Name_numb'].replace('VER_1', 'VER_33')
    fast_f1.rename(columns={'LapStartDate': 'date', 'LapNumber': 'lap', 'Name_numb': 'driverId'}, inplace=True)
    fast_f1.drop(columns=['Driver', 'DriverNumber'], inplace=True)

    # Merge datasets
    df = pd.merge(df, fast_f1, how='cross', on=['lap', 'date', 'driverId'])

    print('✅ Dataframes merged')

    return df

def add_data_fastf1():
     """Getting the data from fastf1 for laps and weather and merge it to our dataset"""
    # GET DATA
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    df2018 = pd.read_csv(os.path.join(root_dir,"raw_data","fastf1", "fastf1_laps_2018_2021.csv"))
    df2022 = pd.read_csv(os.path.join(root_dir,"raw_data","fastf1", "fastf1_laps_2022.csv"))
    df2023 = pd.read_csv(os.path.join(root_dir,"raw_data","fastf1", "fastf1_laps_2023.csv"))
    df2024 = pd.read_csv(os.path.join(root_dir,"raw_data","fastf1", "fastf1_laps_2024.csv"))
    df = pd.concat([df2018, df2022, df2023, df2024], axis=0) # On concatène les données
    weather2018 = pd.read_csv(os.path.join(root_dir,"raw_data","fastf1", "fastf1_weather_2018_2021.csv"))
    weather2022 = (os.path.join(root_dir,"raw_data","fastf1", "fastf1_weather_2022.csv"))
    weather2023 = (os.path.join(root_dir,"raw_data","fastf1", "fastf1_weather_2023.csv"))
    weather2024 = (os.path.join(root_dir,"raw_data","fastf1", "fastf1_weather_2024.csv"))
    weather = pd.concat([weather2018, weather2022, weather2023, weather2024], axis=0) # On concatène les données

    # Rearranging dataset df
    df = df[['Driver', 'DriverNumber', 'LapNumber', 'Stint', 'Compound', 'TyreLife', 'FreshTyre', 'LapStartDate', 'Time']]
    df["Name_numb"] = df["Driver"].astype(str) + "_" + df["DriverNumber"].astype(str)
    df['LapStartDate'] = df['LapStartDate'].str.slice(0, 10)
    df['Name_numb'] = df['Name_numb'].replace('VER_1', 'VER_33')
    df.rename(columns={'LapStartDate': 'date', 'LapNumber': 'lap', 'Name_numb': 'driverId'}, inplace=True)
    df.drop(columns=['Driver', 'DriverNumber'], inplace=True)
    df['Time'] = df['Time'].str.slice(7, 15)
    df = df.dropna(subset=['date'])

    # Rearranging dataset weather
    weather.rename(columns={"Unnamed: 0": "Prelev", "Time": "Timestamp"}, inplace=True)
    weather['Timestamp'] = weather['Timestamp'].str.slice(7, 15)

    # Add date to weather
    dates = df['date'].dropna().unique()
    date_col = []
    date_index = 0

    for i in range(len(weather)):
        current_prelev = weather.iloc[i]['Prelev']
        if current_prelev == 0 and i != 0:
            if date_index < len(dates) - 1:
                date_index += 1
        date_col.append(dates[date_index])
    weather['date'] = date_col

    # Adapting datasets
    weather['Timestamp'] = pd.to_datetime(weather['date'].astype(str) + ' ' + weather['Timestamp'].astype(str))
    weather['date'] = pd.to_datetime(weather['date'])
    df['Time'] = df['date'].astype(str) + ' ' + df['Time'].astype(str)
    df['Time'] = pd.to_datetime(df['Time'])
    df['date'] = pd.to_datetime(df['date'])
    weather = weather.sort_values(['date', 'Timestamp']).reset_index(drop=True)
    df = df.sort_values(['date', 'Time']).reset_index(drop=True)

    # Merging datasets
    fast_f1 = pd.merge_asof(
    df,
    weather,
    left_on="Time",
    right_on="Timestamp",
    by="date",
    direction='backward'
    )
    print('✅ Dataframes merged')


    # Merge datasets
    df = pd.merge(df, fast_f1, how='cross', on=['lap', 'date', 'driverId'])

    print('✅ Dataframes merged')

    return df
