from __future__ import annotations

from pathlib import Path
from autogluon.tabular import TabularPredictor

_MODEL = None
_MODEL_VERSION = None

MODEL_DIR = Path("AutogluonModels/ag-20251119_185801")  # production


def get_model() -> tuple[TabularPredictor, str]:
    # singleton
    global _MODEL, _MODEL_VERSION

    if _MODEL is not None:
        return _MODEL, _MODEL_VERSION

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Nie znaleziono katalogu modelu: {MODEL_DIR}. ")

    _MODEL = TabularPredictor.load(MODEL_DIR, require_version_match=False)

    mtime = int(MODEL_DIR.stat().st_mtime)
    _MODEL_VERSION = f"{MODEL_DIR.name}:{mtime}"

    return _MODEL, _MODEL_VERSION
