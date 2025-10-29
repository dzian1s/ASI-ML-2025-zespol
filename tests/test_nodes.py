import pandas as pd
import numpy as np
import pytest
from asi_ml_2025_zespol.pipelines.data_preparation.nodes import (
    basic_clean,
    train_test_splitting,
)


@pytest.fixture
def sample_data():
    return pd.read_csv("data/01_raw/airbnb_sample.csv")


def test_basic_clean_removes_nans(sample_data):
    cleaned = basic_clean(sample_data)

    # brak NaN
    assert not cleaned.isna().any().any(), "W danych po czyszczeniu pozostały NaN-y"

    # kolumny, które powinny być usunięte
    for col in ["id", "name", "host_id", "host_name", "last_review"]:
        assert col not in cleaned.columns, f"Kolumna {col} nie została usunięta"

    # sprawdź, że dane nie są puste
    assert len(cleaned) > 0, "Po czyszczeniu dataset jest pusty"


def test_train_test_split(sample_data):
    """Sprawdza poprawność podziału danych."""
    cleaned = basic_clean(sample_data)
    X_train, X_test, y_train, y_test = train_test_splitting(
        cleaned, test_size=0.2, random_state=42
    )

    # brak targetu w X
    assert "price" not in X_train.columns
    assert "price" not in X_test.columns

    # poprawna proporcja
    total = len(cleaned)
    expected_test = int(0.2 * total)
    assert (
        abs(len(X_test) - expected_test) <= 1
    ), "Nieprawidłowy rozmiar zbioru testowego"

    # spójność liczby rekordów
    assert (
        len(X_train) + len(X_test) == total
    ), "Rozmiary X_train + X_test ≠ liczba rekordów"
    assert (
        len(y_train) + len(y_test) == total
    ), "Rozmiary y_train + y_test ≠ liczba rekordów"

    # brak przecieku danych
    overlap = pd.merge(X_train, X_test, how="inner")
    assert overlap.empty, "Zbiory train i test mają wspólne rekordy"
