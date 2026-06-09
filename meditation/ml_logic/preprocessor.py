# =============================================================================
# preprocessor.py — Preprocessing des données EEG L-FAME
# =============================================================================

import numpy as np
import pandas as pd
from colorama import Fore, Style
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

# Pré-calculer les masques de fréquences une seule fois
def _build_band_masks(freqs, bands):
    return [(freqs >= lo) & (freqs < hi) for lo, hi in bands]


def preprocess_features_extract_psd(X, bands=PSD_BANDS, fs=FS, nperseg=None, batch_size=64):
    """
    X : (N, C, T)
    returns : (N, B*C)  →  (N, 320)

    batch_size : nb de trials traités à la fois (réduit le pic RAM)
    """
    N, C, T = X.shape
    n_bands = len(bands)
    out = np.empty((N, n_bands * C), dtype=np.float32)

    # Calculer welch sur un seul trial pour récupérer les fréquences
    freqs, _ = welch(X[0], fs=fs, nperseg=nperseg, axis=-1)
    masks = _build_band_masks(freqs, bands)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        _, psd_batch = welch(X[start:end], fs=fs, nperseg=nperseg, axis=-1)
        # psd_batch : (batch, C, F)

        features = [psd_batch[:, :, mask].mean(axis=-1) for mask in masks]
        out[start:end] = np.concatenate(features, axis=1)

        del psd_batch  # libère explicitement le batch

    return out
