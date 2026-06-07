import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from meditation.ml_logic.data_load import load_data
from meditation.interface.main import pred_task1

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def root():
    return {"greeting": "Hello WORLD!!!!"}


# http://127.0.0.1:8000/predict/task1
"""
Make a single course prediction.
- sujets : liste de int entre 1 et 74
- labels : liste de str parmi {'Medita', 'slMedita', 'restCE01', 'restCE02', 'restOE'}
- sessions : liste de str parmi {'premedita', 'posmedita'}
- start correspond au début de la 1ère séquence, par défaut à 0
- window_size correspond à la largueur d'une sequence, 1000 points (4s) par défaut
- step correspond au décalage de chaque window
- split permet d'assigner automatiquement des sujets (si sujets=None),
    parmi {'train', 'val', 'test', 'train_small', 'val_small', 'test_small'}
"""
@app.get("/predict/task1")
def pred_task1(
            sujets=[1],
            labels=['Medita', 'slMedita', 'restCE01', 'restCE02', 'restOE'],
            sessions=['premedita'],
            window_size=1000,
            step=1000,
            start=0,
            split=None,
            root=None
            )-> np.ndarray:

    # 2. Build X_pred as a single-row DataFrame

    X_pred = load_data( sujets=sujets,
                        labels=labels,
                        sessions=sessions,
                        window_size=window_size,
                        step=step,
                        start=start,
                        split=split,
                        root=Path.cwd()
                          )

    # 4. Predict using the cached model
    y_pred = pred_task1(X_pred)


    # 5. Return JSON response
    return {"fare": float(y_pred[0][0])}
