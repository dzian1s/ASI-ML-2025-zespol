import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


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
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {"RMSE" :np.sqrt(mean_squared_error(y_test, y_pred))}