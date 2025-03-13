import kagglehub
import pandas as pd
import numpy as np
import os
import ast


def kaggle_to_df():
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


def change_driveId(df):
    """Changing the driver IDs to scale the data properly"""

    # Sorting all the driverIds
    sorted_driver_ids = sorted(df["driverId"].unique())
    # Creating an array of all the gaps in the dataset (a gap is calculated when there is a difference > 1 between 2 values)
    gaps = np.diff(sorted_driver_ids)
    # Getting the index of the largest gap of the array
    max_gap_index = np.argmax(gaps)
    # Finding where the gap starts
    gap_start = sorted_driver_ids[max_gap_index]
    # Finding the "real" size of the gap
    largest_gap_size = gaps[max_gap_index] - 1
    # Filling the biggest gap by shifting the largest driverId values backward
    df.loc[df["driverId"] > gap_start, "driverId"] -= largest_gap_size

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


def def_undercut_tentative(df):
    # Créer un dictionnaire des pit stops (raceId, driverId, lap) → pit_duration
    pit_info = df[df['pit_duration'].notna()].set_index(['raceId', 'driverId', 'lap'])['pit_duration'].to_dict()

    def check_undercut_tentative(row):
        if pd.isna(row['pit_duration']):  # Vérifier si le pilote a fait un pit stop
            return False

        race_id, lap, driver_id = row['raceId'], row['lap'], row['driverId']

        # Récupérer la ligne du pilote au tour précédent
        previous_lap = df[(df['raceId'] == race_id) & (df['lap'] == lap - 1) & (df['driverId'] == driver_id)]

        if previous_lap.empty:  # Si pas de données pour le tour précédent, on sort
            return False

        # Récupérer les rivaux du tour précédent sous forme de liste
        previous_rivals = previous_lap.iloc[0]['rivals']

        if not previous_rivals:  # Si la liste des rivaux du tour précédent est vide
            return False

        # Récupérer les rivaux du lap précédent dans le DataFrame
        previous_lap_rivals = df[(df['raceId'] == race_id) & (df['lap'] == lap - 1) & df['driverId'].isin(previous_rivals)]

        if previous_lap_rivals.empty:  # Si aucun rival du tour précédent n'est trouvé
            return False

        # Vérifier si un de ces rivaux a pité aux tours suivants (lap+1 ou lap+2)
        for _, rival_row in previous_lap_rivals.iterrows():
            for next_lap in [lap + 1, lap + 2]:
                if (race_id, rival_row['driverId'], next_lap) in pit_info:
                    return True  # Undercut tenté

        return False  # Si aucun rival du tour précédent n'a pité après

    # Appliquer la fonction à chaque ligne du DataFrame
    df['undercut_tentative'] = df.apply(check_undercut_tentative, axis=1)

    return df  # Retourner le DataFrame modifié


def def_undercut_success(df):
    # Créer des dictionnaires pour récupérer rapidement les informations nécessaires
    pit_info = df[df['pit_duration'].notna()].set_index(['raceId', 'driverId', 'lap'])['pit_duration'].to_dict()
    position_info = df.set_index(['raceId', 'driverId', 'lap'])['position'].to_dict()

    def check_undercut_success(row):
        if not row['undercut_tentative']:  # Vérifier si l'undercut a été tenté
            return False

        race_id, lap, driver_id = row['raceId'], row['lap'], row['driverId']

        # Récupérer la position du pilote au tour précédent
        driver_pos_lap_minus1 = position_info.get((race_id, driver_id, lap - 1), None)

        if driver_pos_lap_minus1 is None:
            return False  # Si pas de données pour le tour précédent, on ne peut pas comparer

        # Récupérer les rivaux du tour précédent sous forme de liste
        previous_lap = df[(df['raceId'] == race_id) & (df['lap'] == lap - 1) & (df['driverId'] == driver_id)]
        if previous_lap.empty:
            return False

        previous_rivals = previous_lap.iloc[0]['rivals']


        if not previous_rivals:  # Si aucun rival au lap précédent
            return False

        # Vérifier si un de ces rivaux a pité aux tours suivants (lap+1 ou lap+2)
        for rival in previous_rivals:
            for rival_lap in [lap + 1, lap + 2]:  # Vérifier après le pit du pilote
                if (race_id, rival, rival_lap) in pit_info:  # Si le rival a fait un pit
                    # Récupérer la position du rival au lap précédent et au lap après son pit
                    rival_pos_lap_minus1 = position_info.get((race_id, rival, lap - 1), None)
                    rival_pos_after_pit = position_info.get((race_id, rival, rival_lap + 1), None)
                    driver_pos_after_rival_pit = position_info.get((race_id, driver_id, rival_lap + 1), None)

                    # Vérifier que toutes les positions existent
                    if rival_pos_lap_minus1 is not None and rival_pos_after_pit is not None and driver_pos_after_rival_pit is not None:
                        # Vérifier que le pilote était derrière avant, et est passé devant après
                        if driver_pos_lap_minus1 > rival_pos_lap_minus1 and driver_pos_after_rival_pit < rival_pos_after_pit:
                            return True  # Undercut réussi

        return False  # Si aucun cas ne valide l'undercut réussi

    # Appliquer la fonction à chaque ligne du DataFrame
    df['undercut_success'] = df.apply(check_undercut_success, axis=1)

    return df  # Retourner le DataFrame modifié


def baseline_small_dataset(df):
    # For our baseline model we keep the 1,500 rows in which there is an undercut tentative
    df = df[df["undercut_tentative"] == True]
    return df


def baseline_data_prep():
    df = kaggle_to_df()
    df = change_driveId(df)
    df = identify_rivals(df)
    df = def_undercut_tentative(df)
    df = def_undercut_success(df)
    df = baseline_small_dataset(df)
    return df


def normal_data_prep():
    df = kaggle_to_df()
    df = change_driveId(df)
    df = identify_rivals(df)
    df = def_undercut_tentative(df)
    df= def_undercut_success(df)
    # TO BE MODIFIED
    return df


if __name__ == '__main__':
    kaggle_to_df()
    change_driveId()
    identify_rivals()
    def_undercut_tentative()
    def_undercut_success()
    baseline_small_dataset()
    baseline_data_prep()
    normal_data_prep()
