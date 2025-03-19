from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
import os
from f1_project.f1_packages.params import *
scaler_path = SCALER_PATH
encoder_path = ENCODER_PATH

def drop_unnecessary_columns(X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> tuple:
    """
    Drop unnecessary columns from both training and testing data.

    Args:
        X_train: Training features DataFrame
        X_test: Testing features DataFrame (optional)

    Returns:
        tuple: (X_train_cleaned, X_test_cleaned) if X_test is provided,
               (X_train_cleaned, None) if X_test is None
    """
    columns_to_drop = ["pit_duration", "rivals", "top_rivals", "raceId", "date", "time"]

    # Drop columns from training data
    X_train_cleaned = X_train.drop(columns=columns_to_drop, errors='ignore')

    # Drop columns from test data if provided
    X_test_cleaned = None
    if X_test is not None:
        X_test_cleaned = X_test.drop(columns=columns_to_drop, errors='ignore')

    return X_train_cleaned, X_test_cleaned

def custom_lap_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Custom encoding for lap column that excludes:
    - Laps 1-10 for each circuit
    - Last 5 laps for each circuit
    """
    # Get the last 5 laps for each circuit
    last_laps = X.groupby('name')['lap'].max().to_dict()
    last_5_laps = {circuit: range(max_lap-4, max_lap+1) for circuit, max_lap in last_laps.items()}

    # Create a mask for laps to encode
    def should_encode_lap(row):
        if row['lap'] <= 10 or row['lap'] in last_5_laps[row['name']]:
            return 'special_lap'
        return str(row['lap'])

    # Create encoded column
    X['lap_encoded'] = X.apply(should_encode_lap, axis=1)
    return X

def create_sklearn_preprocessor(dataset) -> ColumnTransformer:
    """
    Create a sklearn preprocessor without fitting it
    """

    if dataset is None:

        # NUMERICAL FEATURES PIPELINE (Robust Scaling)
        numerical_features = ['lap_time', 'cumul_time', 'cumul_stop']  # Removed pit_duration

        robust_scaler = RobustScaler()
        numerical_pipe = Pipeline([
            ('scaler', robust_scaler)
        ])

        # CATEGORICAL FEATURES PIPELINE (One-Hot Encoding)
        categorical_features = ['position', 'name', 'lap_encoded']

        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        categorical_pipe = Pipeline([
            ('onehot', one_hot_encoder)
        ])

        # COMBINED PREPROCESSOR
        final_preprocessor = ColumnTransformer([
            ('num_preproc', numerical_pipe, numerical_features),
            ('cat_preproc', categorical_pipe, categorical_features)
        ], remainder='passthrough', n_jobs=-1)

    elif dataset == "fastf1":
        # NUMERICAL FEATURES PIPELINE (Robust Scaling)
        numerical_features = ['lap_time', 'cumul_time', 'cumul_stop']  # Removed pit_duration

        robust_scaler = RobustScaler()
        numerical_pipe = Pipeline([
            ('scaler', robust_scaler)
        ])

        # NUMERICAL FEATURES PIPELINE (MinMax Scaling)
        numerical_features2 = ["TyreLife", "TrackTemp"]

        min_max_scaler = MinMaxScaler()
        numerical_pipe2 = Pipeline([
            ('scaler', min_max_scaler)
        ])

        # CATEGORICAL FEATURES PIPELINE (One-Hot Encoding)
        categorical_features = ['position', 'name', 'lap_encoded', 'Compound']

        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        categorical_pipe = Pipeline([
            ('onehot', one_hot_encoder)
        ])

        # COMBINED PREPROCESSOR
        final_preprocessor = ColumnTransformer([
            ('num_preproc1', numerical_pipe, numerical_features),
            ('num_preproc2', numerical_pipe2, numerical_features2),
            ('cat_preproc', categorical_pipe, categorical_features)
        ], remainder='passthrough', n_jobs=-1)

    else:
        raise ValueError(f"Invalid dataset type: {dataset}. Expected None or 'fastf1'.")

    return final_preprocessor


def binary_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Converting FreshTyre and Rainfall columns in binary
    """
    # Convert to boolean first to handle both strings and actual booleans
    X['Rainfall_binary'] = X['Rainfall'].astype(str).map({'True': 1, 'False': 0})

    # Drop original column
    X.drop(columns=["Rainfall"], inplace=True)

    # Rename the binary column back to original name
    X.rename(columns={'Rainfall_binary': 'Rainfall'}, inplace=True)

    return X


def preprocess_features(dataset, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Preprocess features for both training and testing data.
    Fits the preprocessor on training data and transforms both training and testing data.

    Args:
        X_train: Training features DataFrame
        X_test: Testing features DataFrame

    Returns:
        tuple: (X_train_processed, X_test_processed)
    """
    print("\n🚀 Starting preprocessing pipeline...")
    print(f"Initial shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Drop unnecessary columns
    print("\n1️⃣ Dropping unnecessary columns...")
    X_train_cleaned, X_test_cleaned = drop_unnecessary_columns(X_train, X_test)
    print(f"Shapes after dropping columns - X_train: {X_train_cleaned.shape}, X_test: {X_test_cleaned.shape}")
    print(f"Columns dropped: {set(X_train.columns) - set(X_train_cleaned.columns)}")

    if dataset == "fastf1":
        # Binarizing the columns
        print("\n2️⃣ Transforming columns in binary...")
        X_train_cleaned = binary_encoding(X_train_cleaned)
        X_test_cleaned = binary_encoding(X_test_cleaned)
        print("✅ Binary transformation completed")

    # Apply custom lap encoding to both train and test
    print("\n3️⃣ Applying custom lap encoding...")
    X_train_cleaned = custom_lap_encoding(X_train_cleaned)
    X_test_cleaned = custom_lap_encoding(X_test_cleaned)
    print("✅ Lap encoding completed")
    print(f"Unique lap encoded values in train: {X_train_cleaned['lap_encoded'].nunique()}")

    # Create and fit preprocessor on training data
    print("\n4️⃣ Creating and fitting preprocessor...")
    preprocessor = create_sklearn_preprocessor(dataset)

    # Check if saved preprocessor exists
    if os.path.exists(scaler_path) and os.path.exists(encoder_path):
        print("✅ Loading existing preprocessor from:", scaler_path)
        preprocessor = joblib.load(scaler_path)
    else:
        print("✅ Fitting new preprocessor...")
        preprocessor.fit(X_train_cleaned)
        print(f"Chemin utilisé : {os.path.abspath(scaler_path)}")
        joblib.dump(preprocessor, scaler_path)
        joblib.dump(preprocessor, encoder_path)
        print("✅ Preprocessor saved to:", scaler_path)

    # Transform both train and test data
    print("\n5️⃣ Transforming data...")
    X_train_processed = preprocessor.transform(X_train_cleaned)
    X_test_processed = preprocessor.transform(X_test_cleaned)

    print("\n✨ Final results:")
    print(f"X_train processed shape: {X_train_processed.shape}")
    print(f"X_test processed shape: {X_test_processed.shape}")
    print("✅ Preprocessing completed successfully!")

    return pd.DataFrame(X_train_processed), pd.DataFrame(X_test_processed)


if __name__ == '__main__':
    preprocess_features(DATASET_CHOICE)
