import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from meditation.ml_logic.preprocessor import preprocess_features_extract_psd
from meditation.ml_logic.data_load import load_data
from meditation.params import *
from meditation.ml_logic.registry import save_model
from meditation.ml_logic.registry import load_model


def train_task1() -> float:
    """
    - implementation for task1
    - Download processed data
    - Train on the preprocessed dataset
    - Store training results and model weights

    Return accuracy as a float
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)




    # Create (X_train_processed, y_train, X_val_processed, y_val)
    X ,y = load_data(sujets=[1], labels=['Medita', 'restCE01', 'restOE','slMedita', 'restCE02'], sessions=['posmedita'], window_size=1000, start=4, root=Path.cwd())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_preprocess = preprocess_features_extract_psd(X_train)



    # Train model using `model.py`
    model = load_model()

    if model is None:
        model =SVC(kernel="linear", C=1.0)

    model.fit(X_preprocess, y_train)


    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)


    X_test_sca= preprocess_features_extract_psd(X_test)

    y_pred= model.predict(X_test_sca)

    accuracy=accuracy_score(y_test, y_pred)

    print()
    print("\n✅ accuracy: ", accuracy, "\n")

    return accuracy, X_test_sca



def pred_task1(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model for task 1
    """

    print("\n⭐️ Use case: predict")




    model = load_model()
    assert model is not None



    X_processed = preprocess_features_extract_psd(X_pred)
    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred

def train_task2():

    """
    implementation de la task2

    """




def train_task3():
    """
    implementation de la task3

    """
