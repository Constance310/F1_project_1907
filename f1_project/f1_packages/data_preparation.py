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

def def_undercut_tentative(row, df):
    if pd.isna(row['pit_duration']):  # Vérifier si le pilote a fait un pit stop
        return False

    # Créer un dictionnaire des pit stops (raceId, driverId, lap) → pit_duration
    pit_info = df[df['pit_duration'].notna()].set_index(['raceId', 'driverId', 'lap'])['pit_duration'].to_dict()

    race_id, lap, driver_id = row['raceId'], row['lap'], row['driverId']

    # Récupérer la ligne du pilote au tour précédent
    previous_lap = df[(df['raceId'] == race_id) & (df['lap'] == lap - 1) & (df['driverId'] == driver_id)]

    if previous_lap.empty:  # Si pas de données pour le tour précédent, on sort
        return False

    # Récupérer les rivaux du lap précédent
    previous_rivals = previous_lap.iloc[0]['rivals']

    # Récupérer les rivaux du lap précédent dans le DataFrame
    previous_lap_rivals = df[(df['raceId'] == race_id) & (df['lap'] == lap - 1)]

    # Filtrer uniquement les pilotes qui sont dans la liste des rivaux du tour précédent
    previous_lap_rivals = previous_lap_rivals[previous_lap_rivals['driverId'].isin(previous_rivals)]

    if previous_lap_rivals.empty:  # Si aucun rival du tour précédent n'est trouvé
        return False

    # Vérifier si un de ces rivaux a pité aux tours suivants (lap+1 ou lap+2)
    for _, rival_row in previous_lap_rivals.iterrows():
        for next_lap in [lap + 1, lap + 2]:
            if (race_id, rival_row['driverId'], next_lap) in pit_info:
                return True  # Undercut tenté

    return False  # Si aucun rival du tour précédent n'a pité après


def def_undercut_success(row, df):
    if not row['undercut_tentative']:  # Vérifier d'abord si l'undercut a été tenté
        return False

    race_id, lap, driver_id = row['raceId'], row['lap'], row['driverId']

    # Créer un dictionnaire des positions (raceId, driverId, lap) → position
    position_info = df.set_index(['raceId', 'driverId', 'lap'])['position'].to_dict()

    # Créer un dictionnaire des pit stops (raceId, driverId, lap) → pit_duration
    pit_info = df[df['pit_duration'].notna()].set_index(['raceId', 'driverId', 'lap'])['pit_duration'].to_dict()

    # Récupérer la ligne du pilote au tour précédent (lap - 1)
    previous_lap = df[(df['raceId'] == race_id) & (df['lap'] == lap - 1) & (df['driverId'] == driver_id)]

    if previous_lap.empty:  # Si pas de données pour le tour précédent, on retourne False
        return False

    # Récupérer les rivaux du tour précédent
    previous_rivals = previous_lap.iloc[0]['rivals']


    # Récupérer la position du pilote au lap - 1
    driver_position_lap_minus1 = position_info.get((race_id, driver_id, lap - 1), None)

    if driver_position_lap_minus1 is None:
        return False

    # Parcourir les rivaux du tour précédent
    for rival in previous_rivals:
        # Vérifier si ce rival a fait un pit aux tours suivants (lap+1 ou lap+2)
        for rival_lap in (lap + 1, lap + 2):
            if (race_id, rival, rival_lap) in pit_info:  # Si le rival a pité après le pilote
                # Récupérer la position du rival au lap - 1
                rival_position_lap_minus1 = position_info.get((race_id, rival, lap - 1), None)

                # Vérifier que la position initiale du rival existe
                if rival_position_lap_minus1 is None:
                    continue

                # Vérifier la position du pilote et du rival après le pit du rival (lap+1 ou lap+2)
                for check_lap in (rival_lap + 1, rival_lap + 2):
                    driver_position_after = position_info.get((race_id, driver_id, check_lap), None)
                    rival_position_after = position_info.get((race_id, rival, check_lap), None)

                    # Vérifier que les positions existent
                    if driver_position_after is not None and rival_position_after is not None:
                        if driver_position_after < rival_position_after:  # Si le pilote est passé devant
                            return True

    return False  # Si aucune des conditions n'est remplie


if __name__ == '__main__':
    kaggle_to_df()
    identify_rivals()
    def_undercut_tentative()
    def_undercut_success()
