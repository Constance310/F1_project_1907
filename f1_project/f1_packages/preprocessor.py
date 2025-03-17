from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
import os
from params import SCALER_PATH, ENCODER_PATH

scaler_path = SCALER_PATH
encoder_path = ENCODER_PATH

def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    def create_sklearn_preprocessor() -> ColumnTransformer:
        """
        Scikit-learn pipeline that transforms the dataset by scaling numerical features
        and applying one-hot encoding to categorical features.
        """

        # NUMERICAL FEATURES PIPELINE (Robust Scaling)
        numerical_features = ['lap_time', 'cumul_time', 'cumul_stop', 'pit_duration']
        scaler_path = "robust_scaler.pkl"

        if os.path.exists(scaler_path):
            robust_scaler = joblib.load(scaler_path)
            print("✅ Loaded existing RobustScaler")
        else:
            robust_scaler = RobustScaler()
            robust_scaler.fit(X[numerical_features])
            joblib.dump(robust_scaler, scaler_path)
            print("✅ Trained and saved RobustScaler")

        numerical_pipe = Pipeline([
            ('scaler', robust_scaler)
        ])

        # CATEGORICAL FEATURES PIPELINE (One-Hot Encoding)
        categorical_features = ['lap', 'position', 'name']
        encoder_path = "one_hot_encoder.pkl"

        if os.path.exists(encoder_path):
            one_hot_encoder = joblib.load(encoder_path)
            print("✅ Loaded existing OneHotEncoder")
        else:
            one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            one_hot_encoder.fit(X[categorical_features])
            joblib.dump(one_hot_encoder, encoder_path)
            print("✅ Trained and saved OneHotEncoder")

        categorical_pipe = Pipeline([
            ('onehot', one_hot_encoder)
        ])

        # COMBINED PREPROCESSOR
        final_preprocessor = ColumnTransformer([
            ('num_preproc', numerical_pipe, numerical_features),
            ('cat_preproc', categorical_pipe, categorical_features)
        ], remainder='passthrough', n_jobs=-1)

        return final_preprocessor

    print("\nPreprocessing features...")
    preprocessor = create_sklearn_preprocessor()
    X_processed = preprocessor.fit_transform(X)
    print("✅ X_processed, with shape", X_processed.shape)

    return pd.DataFrame(X_processed)
