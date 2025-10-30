import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import wandb
import yaml
from pathlib import Path


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
