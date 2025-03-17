import pandas as pd
from f1_project.f1_packages.params import *





def def_undercut_tentative(df):
    # Créer un dictionnaire des pit stops (raceId, driverId, lap) → pit_duration
    pit_info = df[(df['pit_duration'].notna()) & (df['pit_duration'] != 0)].set_index(['raceId', 'driverId', 'lap'])['pit_duration'].to_dict()

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

def top_X_rivals_columns(df : pd.DataFrame):
    # Extraire les rivaux et exploser la liste
    df_rivals = df[['driverId', 'rivals']].copy()
    df_exploded = df_rivals.explode('rivals')

    # Compter les occurrences de chaque rival
    rival_counts = df_exploded.groupby(['driverId', 'rivals']).size().reset_index(name='count')

    # Trier et garder les X rivaux les plus fréquents
    top_X_rivals = rival_counts.sort_values(['driverId', 'count'], ascending=[True, False])
    top_X_rivals = top_X_rivals.groupby('driverId').head(TOP_RIVALS)

    # Transformer en liste pour chaque driver
    top_X_rivals_list = top_X_rivals.groupby('driverId')['rivals'].apply(list).reset_index()
    top_X_rivals_list.rename(columns={'rivals': 'top_rivals'}, inplace=True)

    # Merge avec le DataFrame original
    df = df.merge(top_X_rivals_list, on='driverId', how='left')

    # Remplacer NaN par des listes vides (évite les erreurs d'itération)
    df['top_rivals'] = df['top_rivals'].apply(lambda x: x if isinstance(x, list) else [])

    # Créer les 5 colonnes pour les rivaux du top 5
    for i in range(TOP_RIVALS):
        col_name = f'top_rival_{i+1}'
        df[col_name] = df.apply(lambda row: int(row['top_rivals'][i] in row['rivals']) if i < len(row['top_rivals']) else 0, axis=1)

    return df

def add_pit_column(df, numero_pit: int):
    # Sélectionner les premiers pit stops (cumul_stop == 1)
    df_pit = df[['name', 'lap']][df.cumul_stop == numero_pit]

    # Calculer le tour moyen de pit1 par circuit
    pit_avg = df_pit.groupby('name')['lap'].median().reset_index(name=f'pit{numero_pit}')
    pit_avg[f'pit{numero_pit}'] = pit_avg[f'pit{numero_pit}'].round().astype(int)

    # Fusionner cette information avec le DataFrame principal
    df = df.merge(pit_avg, on='name', how='left')

    for i in range(numero_pit) :
        df = add_pit_column(df, i+1)

    return df


if __name__ == '__main__':
    def_undercut_tentative()
    def_undercut_success()
