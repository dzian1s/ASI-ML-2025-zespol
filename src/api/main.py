from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from src.api.model_loader import get_model
from src.db.database import init_db

app = FastAPI()

predictor = None
model_version = "unknown"


class Features(BaseModel):
    neighbourhood_group: int
    neighbourhood: int
    latitude: int
    longitude: int
    room_type: int
    minimum_nights: int
    number_of_reviews: int
    reviews_per_month: float
    calculated_host_listings_count: int
    availability_365: int


class Prediction(BaseModel):
    prediction: float
    model_version: str


predictor = None
model_version = "unknown"


@app.on_event("startup")
def startup():
    init_db()
    global predictor, model_version
    predictor, model_version = get_model()


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": model_version}


@app.post("/predict", response_model=Prediction)
def predict(payload: Features):
    try:
        X = pd.DataFrame([payload.model_dump()])
        y_pred = predictor.predict(X)
        return {"prediction": float(y_pred.iloc[0]), "model_version": model_version}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
