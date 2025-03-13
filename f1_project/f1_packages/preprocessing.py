from sklearn.preprocessing import OneHotEncoder,RobustScaler
import numpy as np

def preproc_ohe_scaling(df):
    # Instantiate Robust Scaler
    rb_scaler = RobustScaler()
    # Fit the scaler to lap_time, cumul_time, cumul_stop et pit_duration
    rb_scaler.fit(df[['lap_time', 'cumul_time', 'cumul_stop', 'pit_duration']])
    # Scale / transform lap_time
    df[['lap_time', 'cumul_time', 'cumul_stop', 'pit_duration']] = rb_scaler.transform(df[['lap_time', 'cumul_time', 'cumul_stop', 'pit_duration']])

    #One hot encoding of the lap
    # Instantiate the OneHotEncoder
    ohe = OneHotEncoder(sparse_output=False)
    # Fit encoder
    ohe.fit(df[['lap']])
    # Transform the current "lap" column
    df[ohe.get_feature_names_out()] = ohe.transform(df[['lap']])
    # Drop the column "lap" which has been encoded
    df = df.drop(columns=["lap"])

    #One hot encoding of the position
    # Fit encoder
    ohe.fit(df[['position']])
    # Transform the current "position" column
    df[ohe.get_feature_names_out()] = ohe.transform(df[['position']])
    # Drop the column "position" which has been encoded
    df = df.drop(columns=["position"])

    #One hot encoding of the name of the circuit
    # Fit encoder
    ohe.fit(df[['name']])
    # Transform the current "name" column
    df[ohe.get_feature_names_out()] = ohe.transform(df[['name']])
    # Drop the column "name" which has been encoded
    df = df.drop(columns=["name"])

    return df

if __name__ == '__main__':
   preproc_ohe_scaling()
