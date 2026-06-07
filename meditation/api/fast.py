import numpy as np
import pandas as pd
from fastapi import FastAPI,Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import ast
from typing import List, Optional

from meditation.ml_logic.data_load import load_data
from meditation.interface.main import pred_task1 as model_pred_task1




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


# http://127.0.0.1:8000/predict/task1?sujets=1&labels=Medita&labels=slMedita&labels=restCE01&sessions=premedita&window_size=1000&step=1000&start=0
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
@app.get("/predict/task1", response_model=None)
def pred_task1(
    sujets: List[str] = Query(default=[]),
    labels: List[str] = Query(default=[]),
    sessions: List[str] = Query(default=[]),
    window_size: int = 1000,
    step: int = 1000,
    start: int = 0,
    split: Optional[str] = None
):
    X_test, y_test = load_data(
        sujets=sujets,
        labels=labels,
        sessions=sessions,
        window_size=window_size,
        step=step,
        start=start,
        split=split,
        root=Path.cwd()
    )


    y_pred = model_pred_task1(X_test)

    return {"predictions": y_pred.tolist(),
            "test": y_test.tolist()}
