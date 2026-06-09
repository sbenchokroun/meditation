import glob
import os
import time
import pickle
import joblib

from typing import Literal
from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage
from meditation.params import *




ModelType = Literal["inter_task1", "inter_task2", "intra_task1"]


def save_model(model: keras.Model = None, model_type: ModelType = "inter_task1") -> None:
    """
    Persist trained model locally at:
      f"{LOCAL_REGISTRY_PATH}/models/{model_type}/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist in GCS at "models/{model_type}/{timestamp}.h5"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", model_type, f"{timestamp}.h5")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    print(f"✅ Model saved locally ({model_type})")

    if MODEL_TARGET == "gcs":
        model_filename = model_path.split("/")[-1]
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(f"models/{model_type}/{model_filename}")
        blob.upload_from_filename(model_path)

        print(f"✅ Model saved to GCS ({model_type})")

    return None


def load_model(model_type: ModelType = "inter_task1") -> keras.Model:
    """
    Return the latest saved model for the given model_type:
    - locally: from {LOCAL_REGISTRY_PATH}/models/{model_type}/
    - or from GCS if MODEL_TARGET=='gcs'

    Return None if no model found.
    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest {model_type} model from local registry..." + Style.RESET_ALL)

        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models", model_type)
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
        latest_model = joblib.load(most_recent_model_path_on_disk)

        print(f"✅ Model loaded from local disk ({model_type})")
        return latest_model

    elif MODEL_TARGET == "gcs":
        print(Fore.BLUE + f"\nLoad latest {model_type} model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(GCS_BUCKET_NAME).list_blobs(prefix=f"models/{model_type}"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
            os.makedirs(os.path.dirname(latest_model_path_to_save), exist_ok=True)
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = keras.models.load_model(latest_model_path_to_save)

            print(f"✅ Latest {model_type} model downloaded from GCS")
            return latest_model
        except:
            print(f"\n❌ No model found in GCS for type '{model_type}'")
            return None
