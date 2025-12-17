import sys
import os
from pathlib import Path
import pytest
import pandas as pd
from unittest.mock import patch
from fastapi.testclient import TestClient

src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from api.main import app

client = TestClient(app)


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_validation_error():
    payload = {"neighbourhood_group": "tekst_zamiast_liczby"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@patch("api.main.predictor")
@patch("api.main.save_prediction")
def test_predict_integration(mock_save, mock_predictor):
    mock_predictor.predict.return_value = pd.Series([123.45])

    valid_payload = {
        "neighbourhood_group": 1,
        "neighbourhood": 45,
        "latitude": 40720000,
        "longitude": -73990000,
        "room_type": 1,
        "minimum_nights": 1,
        "number_of_reviews": 1,
        "reviews_per_month": 1.0,
        "calculated_host_listings_count": 1,
        "availability_365": 1
    }
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == 123.45
    assert mock_save.called