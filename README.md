# New York Airbnb - ASI Project

## Zrodło danych:
https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
### Licencja:
CC0: Public Domain
https://creativecommons.org/publicdomain/zero/1.0/
### Data pobrania:
08.10.2025


## Metryka:
RMSE(Root Mean Square Error)

## Weights&Biases

## Projekt
https://wandb.ai/s27335-polsko-japo-ska-akademia-technik-komputerowych/asi-ml-2025-zespol

### Run
https://wandb.ai/s27335-polsko-japo-ska-akademia-technik-komputerowych/asi-ml-2025-zespol/runs/p4py0hol?nw=nwusers25282

### Dla serwera

uvicorn src.api.main:app --reload --port 8000

### docs
http://127.0.0.1:8000/docs

### test health
```bash
curl http://127.0.0.1:8000/healthz
```


### predykcja 
```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "neighbourhood_group": 1,
           "neighbourhood": 45,
           "latitude": 40720000,
           "longitude": -73990000,
           "room_type": 2,
           "minimum_nights": 3,
           "number_of_reviews": 128,
           "reviews_per_month": 1.35,
           "calculated_host_listings_count": 2,
           "availability_365": 210
         }'
```

### Kedro Quickstart
Aby uruchomić projekt:

```bash
kedro run

### Wymagania

- Python 3.11
- Kedro 1.0.0
- Autogluon 1.4.0
- Kedro-Datasets 2.1.0
- scikit-learn, pandas, numpy
```