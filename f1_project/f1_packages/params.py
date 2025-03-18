import os

START_LAP_RIVALS = int(10)
TIME_RIVALS = int(5)
NUMBER_PIT_MAX = int(4)
LAP_DURATION_MAX = int(180000)
PIT_DURATION_MAX = int(55000)
SCALER_PATH = "models/robust_scaler.pkl"
ENCODER_PATH = "models/one_hot_encoder.pkl"
MODEL_VERSION= "light"
TOP_RIVALS = 4
MODEL_TARGET = os.environ.get("MODEL_TARGET")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")
NUMBER_OF_STOPS = 3
