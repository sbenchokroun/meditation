# =============================================================================
# preprocessor.py — Preprocessing des données EEG L-FAME
# =============================================================================

import numpy as np
import pandas as pd
from colorama import Fore, Style
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


# Z-score global calculé sur le dataset L-FAME (ml_preproc_data)

EEG_MAX  = 160


def preprocess_features(X) -> np.ndarray:
    print(Fore.BLUE + "\nPreprocessing features..." + Style.RESET_ALL)

    # Drop label column if present
    if isinstance(X, pd.DataFrame) and "label" in X.columns:
        X = X.drop(columns=["label"]).values
    elif isinstance(X, pd.DataFrame):
        X = X.values

    X_processed = (X) / EEG_MAX

    print("✅ X_processed, with shape", X_processed.shape)
    return X_processed
