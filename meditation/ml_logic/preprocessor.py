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
EEG_MEAN = -0.05
EEG_STD  = 13.22


def preprocess_features(X: pd.DataFrame) -> np.ndarray:

    def create_sklearn_preprocessor() -> ColumnTransformer:
        """
        Pipeline scikit-learn qui applique un z-score sur tous les canaux EEG.
        Stateless : fit_transform() == transform().
        """
        eeg_columns = [col for col in X.columns if col != "label"]

        # EEG PIPE — z-score : (x - mean) / std
        eeg_pipe = FunctionTransformer(
            lambda arr: (arr - EEG_MEAN) / EEG_STD
        )

        final_preprocessor = ColumnTransformer(
            [
                ("eeg_zscore", eeg_pipe, eeg_columns),
            ],
            remainder="drop",   # on exclut la colonne "label"
            n_jobs=-1,
        )

        return final_preprocessor

    print(Fore.BLUE + "\nPreprocessing features..." + Style.RESET_ALL)

    preprocessor = create_sklearn_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    print("✅ X_processed, with shape", X_processed.shape)

    return X_processed
