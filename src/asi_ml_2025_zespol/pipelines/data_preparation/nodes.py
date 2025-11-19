import random
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import wandb
import yaml
from pathlib import Path
import logging
from autogluon.tabular import TabularPredictor


def load_raw():
    return pd.read_csv("data/01_raw/airbnb_sample.csv")


def basic_clean(data):
    data_dropped_nulls = data.dropna()

    data_dropped = data_dropped_nulls.drop(
        columns=["id", "name", "host_id", "host_name", "last_review"]
    )
    le_neigh = LabelEncoder()
    data_dropped["neighbourhood"] = le_neigh.fit_transform(
        data_dropped["neighbourhood"]
    )

    le_neigh_group = LabelEncoder()
    data_dropped["neighbourhood_group"] = le_neigh_group.fit_transform(
        data_dropped["neighbourhood_group"]
    )

    le_room = LabelEncoder()
    data_dropped["room_type"] = le_room.fit_transform(data_dropped["room_type"])

    return data_dropped


def train_test_splitting(data, test_size, random_state):
    X = data.drop(columns=["price"])
    y = data["price"].to_frame(name="target")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_baseline(X_train, y_train, n_estimators, random_state):
    credentials_path = Path("conf/local/credentials.yml")
    with open(credentials_path, "r") as f:
        creds = yaml.safe_load(f)
    wandb_creds = creds["wandb"]["api_key"]

    wandb.login(key=wandb_creds)
    wandb.init(
        project="asi-ml-2025-zespol",
        job_type="train",
        reinit=True,
        config={"n_estimators": n_estimators, "random_state": random_state},
    )

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    for i in range(n_estimators):
        model.n_estimators = i + 1
        model.fit(X_train, y_train)

        y_pred = model.predict(X_train)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))

        wandb.log({"train_rmse": rmse})

    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {"RMSE": np.sqrt(mean_squared_error(y_test, y_pred))}


logger = logging.getLogger(__name__)


def train_autogluon(X_train, y_train, params):
    seed = params["random_seed"]

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)

    run = wandb.init(
        project="asi-ml-2025-zespol", job_type="ag-train", reinit=True, config=params
    )

    train_data = X_train.copy()
    train_data[params["label"]] = y_train

    start = time.time()

    predictor = TabularPredictor(
        label=params["label"],
        problem_type=params["problem_type"],
        eval_metric=params["eval_metric"],
    ).fit(
        train_data=train_data,
        presets=params["presets"],
        time_limit=params["time_limit"],
    )

    train_time = time.time() - start

    wandb.log({"train_time_s": train_time})

    return predictor


def evaluate_autogluon(predictor, X_test: pd.DataFrame, y_test: pd.Series):
    predictions = predictor.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    logger.info(f"AutoGluon RMSE: {rmse}")

    # feature importance
    try:
        predictor.feature_importance(X_test)
    except Exception:
        logger.warning("Feature importance unavailable for some models.")

    wandb.log({"rmse": rmse})

    return {"rmse": float(rmse)}


def save_best_model(predictor):
    predictor.save("data/06_models/ag_production.pkl")
    return "data/06_models/ag_production.pkl"


def log_autogluon_artifact(predictor):
    model_path = "data/06_models/ag_production.pkl"
    predictor.save(model_path)

    art = wandb.Artifact("ag_model", type="model")
    art.add_file(model_path)

    wandb.log_artifact(art, aliases=["candidate"])

    return None
