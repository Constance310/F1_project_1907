import os

START_LAP_RIVALS = os.environ.get('START_LAP_RIVALS')
TIME_RIVALS = os.environ.get('TIME_RIVALS')
NUMBER_PIT_MAX = os.environ.get('NUMBER_PIT_MAX')
LAP_DURATION_MAX = os.environ.get('LAP_DURATION_MAX')
PIT_DURATION_MAX = os.environ.get('PIT_DURATION_MAX')
SCALER_PATH = "models/robust_scaler.pkl"
ENCODER_PATH = "models/one_hot_encoder.pkl"
MODEL_VERSION=["light", "normal"]
