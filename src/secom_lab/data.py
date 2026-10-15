"""Dataset acquisition, provenance and strict inference contracts."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
SENSORS = [f"sensor_{i:03d}" for i in range(590)]  # zero-based raw column positions


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acquire(destination: Path, archive: Path | None = None) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    if archive is None:
        with urlopen(DATA_URL, timeout=60) as response:
            content = response.read(20_000_001)
        if len(content) > 20_000_000:
            raise ValueError("Dataset archive exceeds size limit")
        archive = destination / "secom.zip"
        archive.write_bytes(content)
    with ZipFile(archive) as zipped:
        # Extract known filenames only; never trust archive paths.
        for name in ("secom.data", "secom_labels.data", "secom.names"):
            (destination / name).write_bytes(zipped.read(name))
    X, y, _ = load_data(destination)
    provenance = {
        "source": DATA_URL, "doi": "10.24432/C54305", "license": "CC BY 4.0",
        "acquisition": "local archive" if archive.parent != destination else "HTTPS download",
        "archive_sha256": sha256(archive),
        "files": {n: sha256(destination / n) for n in ("secom.data", "secom_labels.data", "secom.names")},
        "rows": len(X), "measurement_columns": X.shape[1], "failures": int(y.sum()),
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def load_data(directory: Path):
    X = pd.read_csv(directory / "secom.data", sep=r"\s+", header=None)
    labels = pd.read_csv(directory / "secom_labels.data", sep=r"\s+", header=None)
    if X.shape != (1567, 590) or len(labels) != len(X):
        raise ValueError(f"Unexpected SECOM dimensions: {X.shape}, {len(labels)} labels")
    if set(labels[0].unique()) != {-1, 1}:
        raise ValueError("Unexpected SECOM label encoding")
    X.columns = SENSORS
    X = validate_frame(X, max_rows=2000)
    y = (labels[0] == 1).astype(int)
    if int(y.sum()) != 104:
        raise ValueError("Expected 104 failure labels")
    time = pd.to_datetime(labels[1], format="%d/%m/%Y %H:%M:%S", errors="raise")
    return X, y, time


def validate_frame(frame: pd.DataFrame, max_rows: int = 1000) -> pd.DataFrame:
    if not 1 <= len(frame) <= max_rows:
        raise ValueError(f"Provide 1–{max_rows} rows")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate sensor names")
    missing, extra = set(SENSORS) - set(frame), set(frame) - set(SENSORS)
    if missing or extra:
        raise ValueError(f"Expected exactly 590 named sensors; missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}")
    result = frame.loc[:, SENSORS].apply(pd.to_numeric, errors="raise").astype(float)
    if np.isinf(result.to_numpy()).any():
        raise ValueError("Infinite measurements are invalid; use null for missing values")
    if result.isna().all(axis=1).any():
        raise ValueError("A run cannot have every sensor missing")
    return result


def read_csv(content: bytes) -> pd.DataFrame:
    if len(content) > 10_000_000:
        raise ValueError("CSV must be at most 10 MB")
    header = pd.read_csv(io.BytesIO(content), nrows=0).columns
    # Pandas mangles duplicate column names. Inspect the original header separately.
    import csv
    raw = next(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
    if len(raw) != len(set(raw)):
        raise ValueError("Duplicate sensor names")
    if len(header) != 590:
        raise ValueError("CSV must contain 590 sensor columns")
    return validate_frame(pd.read_csv(io.BytesIO(content)))
