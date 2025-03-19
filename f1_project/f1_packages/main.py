from f1_project.f1_packages.data_preparation import *
from f1_project.f1_packages.data_cleaning import *
from f1_project.f1_packages.params import *
from f1_project.f1_packages.preprocessor import *
from f1_project.f1_packages.utils import *
from f1_project.f1_packages.model import *
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Any as Model


def generation_dataframe(model_type):
    """
    Generate the final dataframe based on the specified model version.

    Args:
        MODEL_VERSION (str): Version of the model to use ('light' or 'normal')

    Returns:
        pd.DataFrame: Processed and prepared dataframe ready for modeling
    """
    print(f"\n🔨 Generating dataframe with {model_type} version...")

    if model_type == "light":
        print("\n1️⃣ Performing light cleaning...")
        # Data cleaning with basic steps
        df_clean = light_cleaning()

        print("\n2️⃣ Preparing data with baseline features...")
        # Data preparation with baseline features
        df_good = baseline_data_prep(df_clean)

        # df_baseline.to_csv("raw_data/df_baseline2.csv", index=False)

    if model_type == "normal":
        print("\n1️⃣ Performing normal cleaning...")
        # Data cleaning with additional steps
        df_clean = normal_cleaning()

        print("\n2️⃣ Preparing data with advanced features...")
        # Data preparation with additional features
        df_good = normal_data_prep(df_clean)

    print(f"\n✅ DataFrame generated successfully! Shape: {df_good.shape}")
    save_to_csv(df_good, "df_good")
    return df_good


def test_train_split(df):
    """
    Split the data into training and testing sets.

    Args:
        df (pd.DataFrame): Input dataframe to split

    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Training and testing splits
    """
    print("\n📊 Splitting data into train and test sets...")
    for i in range(1 , NUMBER_OF_STOPS + 1) :
        #df[f"pit{i}"] = df[f'pit{i}'][df[f'pit{i}'].isna()] = 0
        df[f'pit{i}'].fillna(0, inplace=True)
    # Separate features (X) and target (y)
    print("1️⃣ Separating features and target variables...")
    X = df.drop(columns=["undercut_tentative", "undercut_success"])
    y = df["undercut_success"]
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Split data into train and test sets
    print("\n2️⃣ Performing train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,  # 20% for testing
        random_state=42  # For reproducibility
    )

    # Saving the datasets
    save_to_csv(pd.DataFrame(X_train), "X_train")
    save_to_csv(pd.DataFrame(X_test), "X_test")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    # Returning X and y
    return X_train, X_test, y_train, y_test


def processing(dataset, X_train, X_test):
    """
    Process the features using the preprocessing pipeline.

    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features

    Returns:
        tuple: (X_train_processed, X_test_processed) - Processed features
    """
    print("\n⚙️ Processing features...")
    X_train_processed, X_test_processed = preprocess_features(dataset, X_train, X_test)
    return X_train_processed, X_test_processed


if __name__ == '__main__':
    print("\n🚀 Starting main pipeline...")

    # Generate the dataframe
    print("\nStep 1: Generating DataFrame")
    df = generation_dataframe(MODEL_VERSION)

    # Adding fastf1 dataset
    if DATASET == "fastf1":
        df = add_data_fastf1(df)
    else:
        pass

    # Split into train and test sets
    print("\nStep 2: Splitting Data")
    X_train, X_test, y_train, y_test = test_train_split(df)

    # Process the features
    print("\nStep 3: Processing Features")
    X_train_processed, X_test_processed = processing(DATASET_CHOICE, X_train, X_test)

    # Initialize and train model
    print("\nStep 4: Training Model")
    model = initialize_model('randomforest')  # You can change the model type here
    trained_model = train_model(model, X_train_processed, y_train)

    # Make predictions
    print("\nStep 5: Making Predictions")
    y_pred = pred(trained_model, X_test_processed, y_test)

    print("\n✨ Pipeline completed successfully!")
