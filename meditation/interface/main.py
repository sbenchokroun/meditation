import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from meditation.ml_logic.preprocessor import preprocess_features_extract_psd
from meditation.ml_logic.data_load import load_data, load_train_val_test_one_subject, load_data_task_2
from meditation.params import *
from meditation.ml_logic.registry import save_model
from meditation.ml_logic.registry import load_model


def train_task1_intra() -> float:
    """
    - implementation for task1 en intra avec un seul participant
    - Download processed data
    - Train on the preprocessed dataset
    - Store training results and model weights

    Return accuracy as a float
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)


    # Create (X_train_processed, y_train, X_val_processed, y_val)
    X_train, X_val, X_test, y_train, y_val, y_test = load_train_val_test_one_subject(sujet=1,
                                                                                     labels=['Medita', 'restCE01', 'restOE','slMedita', 'restCE02'],
                                                                                     sessions=['premedita','posmedita'],
                                                                                     window_size=1000, step= 250,
                                                                                     start=4, root=Path.cwd())

    X_preprocess = preprocess_features_extract_psd(X_train)



    # Train model using `model.py`
    model = load_model(model_type="intra_task1")

    if model is None:
        model =SVC(kernel="linear", C=1.0)

    model.fit(X_preprocess, y_train)


    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model,model_type="intra_task1")


    X_val_sca= preprocess_features_extract_psd(X_val)

    y_pred= model.predict(X_val_sca)

    accuracy=accuracy_score(y_val, y_pred)

    print()
    print("\n✅ accuracy: ", accuracy, "\n")

    return accuracy



def train_task1_inter():
    """
    - implementation for task1 en inter avec plusieurs participants
    - Download processed data
    - Train on the preprocessed dataset
    - Store training results and model weights

    Return X_preprocess
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)

    # Create (X_train_processed, y_train, X_val_processed, y_val)

    X_train ,y_train = load_data(sujets=[i for i in range(1, 20) ],
                                labels=['Medita', 'restCE01'],
                                sessions=['posmedita'],
                                window_size=1000, step= 250,
                                start=4, root=Path.cwd())

    X_val ,y_val = load_data_task_2(sujets=[28, 46, 53], labels=['slMedita','Medita'])

    X_preprocess = preprocess_features_extract_psd(X_train)

       # Train model using `model.py`
    model = load_model(model_type="inter_task1")

    if model is None:
        model =SVC(kernel="linear", C=1.0)

    model.fit(X_preprocess, y_train)


    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model,model_type="inter_task1")


    X_val_sca= preprocess_features_extract_psd(X_val)

    y_pred= model.predict(X_val_sca)

    accuracy=accuracy_score(y_val, y_pred)

    print()
    print("\n✅ accuracy: ", accuracy, "\n")

    return accuracy



def train_task2() -> float:

    """
    - implementation for task2 en inter avec plusieurs particiapants
    - Download processed data
    - Train on the preprocessed dataset
    - Store training results and model weights

    Return accuracy as a float
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)

    # Create (X_train_processed, y_train, X_val_processed, y_val)

    X_train,y_train = load_data_task_2(sujets=[i for i in range(1, 40) if i not in (28, 46, 53,43)],
                                       labels=['slMedita','Medita'],
                                       root=Path.cwd())


    X_val ,y_val = load_data_task_2(sujets=[28, 46, 53], labels=['slMedita','Medita'])

    X_preprocess = preprocess_features_extract_psd(X_train)



    # Train model using `model.py`
    model = load_model(model_type="inter_task2")

    if model is None:
        model =SVC(kernel="linear", C=1.0)

    model.fit(X_preprocess, y_train)


    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model,model_type="inter_task2")


    X_val_sca= preprocess_features_extract_psd(X_val)

    y_pred= model.predict(X_val_sca)

    accuracy=accuracy_score(y_val, y_pred)

    print()
    print("\n✅ accuracy: ", accuracy, "\n")

    return accuracy


def pred(X_pred, model) -> np.ndarray:
    """
    Make a prediction using the latest trained model for task
    """

    print("\n⭐️ Use case: predict")

    X_processed = preprocess_features_extract_psd(X_pred)
    y_pred = model.predict(X_processed)
    return y_pred
