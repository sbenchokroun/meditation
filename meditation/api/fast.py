import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


"""
Exemple de methode en get

Make a single course prediction.
Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)

@app.get("/predict")
def predict(
        pickup_datetime: str,  # 2014-07-06 19:18:00
        pickup_longitude: float,    # -73.950655
        pickup_latitude: float,     # 40.783282
        dropoff_longitude: float,   # -73.984365
        dropoff_latitude: float,    # 40.769802
        passenger_count: int
    ):

    # 2. Build X_pred as a single-row DataFrame
    X_pred = pd.DataFrame([{
        "pickup_datetime":    pd.Timestamp(pickup_datetime, tz="US/Eastern"),
        "pickup_longitude":   pickup_longitude,
        "pickup_latitude":    pickup_latitude,
        "dropoff_longitude":  dropoff_longitude,
        "dropoff_latitude":   dropoff_latitude,
        "passenger_count":    passenger_count,
    }])

    # 4. Predict using the cached model
    y_pred = pred(X_pred)


    # 5. Return JSON response
    return {"fare": float(y_pred[0][0])}


@app.get("/")
def root():
    return {"greeting": "Hello"}
"""
