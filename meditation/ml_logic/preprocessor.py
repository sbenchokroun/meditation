# =============================================================================
# preprocessor.py — Preprocessing des données EEG L-FAME
# =============================================================================

import numpy as np
import pandas as pd
from colorama import Fore, Style
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from scipy.signal import welch


FS = 250
PSD_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
#             delta  theta   alpha    beta        gamma

EEG_MAX  = 160


def preprocess_normalisation_features(X) -> np.ndarray:
    """
    Calcule la puissance moyenne par bande fréquentielle et par canal.

    X      : (N, C, T)   après transpose_if_needed
    retour : (N, B*C)  = (N, 320)
    """
    print(Fore.BLUE + "\nPreprocessing features..." + Style.RESET_ALL)

    # Drop label column if present
    if isinstance(X, pd.DataFrame) and "label" in X.columns:
        X = X.drop(columns=["label"]).values
    elif isinstance(X, pd.DataFrame):
        X = X.values

    X_processed = (X) / EEG_MAX

    print("✅ X_processed, with shape", X_processed.shape)
    return X_processed


def preprocess_features_extract_psd(X, bands=PSD_BANDS, fs=FS):
    """
    Calcule la puissance moyenne par bande fréquentielle et par canal.

    X      : (N, C, T)   après transpose_if_needed
    retour : (N, B*C)  = (N, 320)
    """
    print(Fore.BLUE + "\nPreprocessing features..." + Style.RESET_ALL)
    X=transpose_if_needed(X)
    freqs, psd = welch(X, fs=fs, axis=-1)          # (N, C, F)
    features = [
        psd[:, :, (freqs >= lo) & (freqs < hi)].mean(axis=-1)
        for lo, hi in bands                         # (N, C) par bande
    ]
    print("✅ X_processed, with shape", features.shape)
    return np.concatenate(features, axis=1)         # (N, 320)


def transpose_if_needed(X):
    """(N, T, C) → (N, C, T)  """
    if X.shape[1] != 64:
        return X.transpose(0, 2, 1)
    return X
