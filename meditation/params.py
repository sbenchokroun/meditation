import os
import numpy as np
from pathlib import Path

##################  VARIABLES  ##################
TRAIN = np.arange(1,51)
VAL = np.arange(51,61)
TEST = np.arange(61,75)

TRAIN_SMALL = np.arange(1,21)
VAL_SMALL = np.arange(21,26)
TEST_SMALL = np.arange(26,31)

##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".meditation", "mlops", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".meditation", "mlops", "training_outputs")

ROOT = Path.cwd().parent


################## VALIDATIONS #################
