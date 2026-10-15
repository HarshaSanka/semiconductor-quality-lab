"""Optional FastAPI service. Run: uvicorn secom_lab.api:app --host 127.0.0.1."""
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .data import SENSORS
from .inference import explain, load_bundle, predict

ARTIFACTS = Path(os.environ.get("SECOM_ARTIFACTS", "artifacts"))
app = FastAPI(title="Semiconductor Quality Lab", version="0.1.0")


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[dict[str, float | None]] = Field(min_length=1, max_length=1000)


@lru_cache(maxsize=1)
def bundle():
    return load_bundle(ARTIFACTS)


@app.get("/health")
def health():
    try:
        model = bundle()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(503, "Train a valid local model first") from error
    return {"status": "ready", "model": model["winner"]}


@app.get("/schema")
def schema():
    return {"sensors": SENSORS, "missing": "JSON null", "positive_class": "fail",
            "scores": "uncalibrated", "max_rows": 1000}


@app.post("/predict")
def predict_runs(batch: Batch):
    try:
        # Validate every row's schema; a union across malformed rows is insufficient.
        if any(set(row) != set(SENSORS) for row in batch.rows):
            raise ValueError("Each row must have exactly the 590 named sensors")
        return {"model": bundle()["winner"], "predictions": predict(bundle(), pd.DataFrame(batch.rows))}
    except (ValueError, TypeError) as error:
        raise HTTPException(422, str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(503, "Train a model first") from error


@app.post("/explain")
def explain_runs(batch: Batch):
    if len(batch.rows) > 10:
        raise HTTPException(422, "Explain at most 10 runs per request")
    if any(set(row) != set(SENSORS) for row in batch.rows):
        raise HTTPException(422, "Each row must have exactly the 590 named sensors")
    try:
        return explain(bundle(), pd.DataFrame(batch.rows), pd.read_csv(ARTIFACTS / "background.csv"))
    except ImportError as error:
        raise HTTPException(503, "Install the full extra for SHAP explanations") from error
    except (ValueError, TypeError) as error:
        raise HTTPException(422, str(error)) from error
