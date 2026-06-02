import glob
import os
import time
import pickle

from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage



def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at

    """


    print("✅ Results saved locally")


def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally

    """



def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model:


    """
