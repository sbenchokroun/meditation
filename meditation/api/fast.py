from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List, Optional
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

from meditation.ml_logic.data_load import load_data
from meditation.interface.main import pred
from meditation.ml_logic.registry import load_model



app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.state.model_intra_task1=load_model('intra_task1')
app.state.model_inter_task1=load_model('inter_task1')
app.state.model_inter_task2=load_model('inter_task2')



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

@app.post("/predict/task1_intra")
def predict_task1_intra(file: UploadFile = File(...)):
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="Le fichier doit être un .npy")

    try:
        arr = np.load(file.file, allow_pickle=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de lire le numpy array: {e}")
    if arr.ndim != 3 or arr.shape[1:] != (1000, 64):
        raise HTTPException(
            status_code=400,
            detail=f"Shape invalide: {arr.shape}. Attendu: (x, 1000, 64)"
        )

    results = pred(arr, app.state.model_intra_task1)

    prediction = int(np.bincount(results).argmax())

    return {"prediction": prediction}

@app.post("/predict/task_inter")
def predict_task1_inter(file: UploadFile = File(...)):
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="Le fichier doit être un .npy")

    try:
        X_test = np.load(file.file, allow_pickle=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de lire le numpy array: {e}")
    if X_test.ndim != 3 or X_test.shape[1:] != (1000, 64):
        raise HTTPException(
            status_code=400,
            detail=f"Shape invalide: {X_test.shape}. Attendu: (x, 1000, 64)"
        )

    results = pred(X_test, app.state.model_inter_task1)

    prediction = int(np.bincount(results).argmax())

    if prediction == 1:
        results_medita = pred(X_test, app.state.model_inter_task2)
        prediction_medita = int(np.bincount(results_medita).argmax())

        return {"prediction" : prediction, "type de meditation" : prediction_medita}

    return {"prediction" : prediction}

@app.post("/predict/task2_inter")
def predict_task2_inter(file: UploadFile = File(...)):
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="Le fichier doit être un .npy")

    try:
        X_test = np.load(file.file, allow_pickle=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de lire le numpy array: {e}")
    if X_test.ndim != 3 or X_test.shape[1:] != (1000, 64):
        raise HTTPException(
            status_code=400,
            detail=f"Shape invalide: {X_test.shape}. Attendu: (x, 1000, 64)"
        )

    results_medita = pred(X_test, app.state.model2)
    prediction_medita = int(np.bincount(results_medita).argmax())
    return {"type de meditation" : prediction_medita}
