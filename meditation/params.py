import os
import numpy as np
from pathlib import Path

##################  VARIABLES  ##################
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")
GCP_REGION = os.environ.get("GCP_REGION")
GCS_BUCKET_NAME=os.environ.get("GCS_BUCKET_NAME")
GCS_DESTINATION_PREFIX=os.environ.get("GCS_DESTINATION_PREFIX")
TRAIN = np.arange(1,51)
VAL = np.arange(51,61)
TEST = np.arange(61,75)

TRAIN_SMALL = np.arange(1,21)
VAL_SMALL = np.arange(21,26)
TEST_SMALL = np.arange(26,31)

##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".meditation", "mlops", "data")
# LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".meditation", "mlops", "training_outputs") # local
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'))


ROOT = Path.cwd().parent


################## VALIDATIONS #################
