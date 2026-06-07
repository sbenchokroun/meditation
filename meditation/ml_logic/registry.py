import glob
import os
import time
import pickle
import joblib

from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage
from meditation.params import *


def save_results(params: dict, metrics: dict) -> None:
    """

    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"

    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")


def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only

    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!

        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None
    return None



def load_model() -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only

    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

        latest_model = joblib.load(most_recent_model_path_on_disk)

        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(GCS_BUCKET_NAME).list_blobs(prefix="model"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = keras.models.load_model(latest_model_path_to_save)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model
        except:
            print(f"\n❌ No model found in GCS bucket {GCS_BUCKET_NAME}")

            return None


def save_scaler(scaler) -> None:
    """
    Persist scaler locally at f"{LOCAL_REGISTRY_PATH}/scalers/{timestamp}.pkl"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "scalers/{timestamp}.pkl"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save scaler locally
    scaler_path = os.path.join(LOCAL_REGISTRY_PATH, "scalers", f"{timestamp}.pkl")
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

    print("✅ Scaler saved locally")

    if MODEL_TARGET == "gcs":
        scaler_filename = scaler_path.split("/")[-1]
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(f"scalers/{scaler_filename}")
        blob.upload_from_filename(scaler_path)

        print("✅ Scaler saved to GCS")

    return None


def load_scaler():
    """
    Return a saved scaler:
    - locally (latest one in alphabetical order)
    - or from GCS if MODEL_TARGET=='gcs'

    Return None (but do not Raise) if no scaler is found
    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest scaler from local registry..." + Style.RESET_ALL)

        local_scaler_directory = os.path.join(LOCAL_REGISTRY_PATH, "scalers")
        local_scaler_paths = glob.glob(f"{local_scaler_directory}/*")

        if not local_scaler_paths:
            print("❌ No scaler found locally")
            return None

        most_recent_scaler_path = sorted(local_scaler_paths)[-1]
        scaler = joblib.load(most_recent_scaler_path)

        print("✅ Scaler loaded from local disk")

        return scaler

    elif MODEL_TARGET == "gcs":
        print(Fore.BLUE + f"\nLoad latest scaler from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(GCS_BUCKET_NAME).list_blobs(prefix="scalers"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_scaler_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
            os.makedirs(os.path.dirname(latest_scaler_path_to_save), exist_ok=True)
            latest_blob.download_to_filename(latest_scaler_path_to_save)

            scaler = joblib.load(latest_scaler_path_to_save)

            print("✅ Latest scaler downloaded from GCS")

            return scaler

        except:
            print(f"\n❌ No scaler found in GCS bucket {GCS_BUCKET_NAME}")
            return None
