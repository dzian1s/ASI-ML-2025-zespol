import json
import datetime as dt
from sqlalchemy import create_engine, text
from src.core.settings import settings

engine = create_engine(
    settings.DATABASE_URL,
    future=True,
    echo=False,
)


def init_db() -> None:
    if engine.url.get_backend_name() == "sqlite":
        stmt = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            payload TEXT,
            prediction REAL,
            model_version TEXT
        )
        """
    else:
        stmt = """
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            ts TIMESTAMP,
            payload JSONB,
            prediction DOUBLE PRECISION,
            model_version TEXT
        )
        """
    with engine.begin() as conn:
        conn.execute(text(stmt))


def save_prediction(
    payload: dict,
    prediction: float | int,
    model_version: str,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO predictions (ts, payload, prediction, model_version)
                VALUES (:ts, :payload, :prediction, :model_version)
                """
            ),
            {
                "ts": dt.datetime.utcnow().isoformat(),
                "payload": json.dumps(payload),
                "prediction": float(prediction),
                "model_version": model_version,
            },
        )
