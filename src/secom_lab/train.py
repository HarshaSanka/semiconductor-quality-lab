"""Auditable experiment runner. Only the CV winner is scored on the final holdout."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import sqlite3
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split

from .data import SENSORS, load_data, sha256
from .evaluation import bootstrap_intervals, choose_threshold, metrics
from .modeling import build_pipeline, experiments, fit_pipeline


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(data_dir=Path("data/raw"), output=Path("artifacts"), profile="core", seed=42, folds=5,
        tracking_uri=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configs = experiments(profile)
    X, y, timestamps = load_data(Path(data_dir))
    dev, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    cv = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X.iloc[dev], y.iloc[dev]))
    split_manifest = {"seed": seed, "development": dev.tolist(), "test": test.tolist(),
                      "folds": [{"train": dev[a].tolist(), "validation": dev[b].tolist()} for a, b in cv]}
    write_json(output / "split_manifest.json", split_manifest)
    summary, oof_by_name, fold_rows = [], {}, []
    mlflow = None
    if tracking_uri:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("secom-quality-benchmark")
    for name, (model, selection) in configs.items():
        start = time.perf_counter()
        print(f"Running {name}", flush=True)
        oof = np.zeros(len(dev))
        fold_ap = []
        for fold, (a, b) in enumerate(cv):
            pipeline = fit_pipeline(build_pipeline(model, selection, seed), X.iloc[dev[a]], y.iloc[dev[a]])
            scores = pipeline.predict_proba(X.iloc[dev[b]])[:, 1]
            oof[b] = scores
            row = {"experiment": name, "fold": fold, **metrics(y.iloc[dev[b]], scores)}
            fold_rows.append(row)
            fold_ap.append(row["average_precision"])
        row = {"experiment": name, "model": model, "selection": selection,
               "cv_ap_mean": float(np.mean(fold_ap)), "cv_ap_std": float(np.std(fold_ap, ddof=1)),
               "oof_average_precision": metrics(y.iloc[dev], oof)["average_precision"],
               "seconds": time.perf_counter() - start}
        summary.append(row)
        oof_by_name[name] = oof
        if mlflow:
            with mlflow.start_run(run_name=name):
                mlflow.log_params({"seed": seed, "folds": folds, "profile": profile,
                                   "model": model, "selection": selection,
                                   "data_sha256": sha256(Path(data_dir) / "secom.data")})
                mlflow.log_metrics({k: row[k] for k in ("cv_ap_mean", "cv_ap_std", "oof_average_precision")})
                mlflow.log_dict({"pipeline": repr(build_pipeline(model, selection, seed))}, "pipeline.json")
                mlflow.log_artifact(str(output / "split_manifest.json"))
    leaderboard = pd.DataFrame(summary).sort_values(["cv_ap_mean", "experiment"], ascending=[False, True])
    winner = leaderboard.iloc[0]["experiment"]
    model, selection = configs[winner]
    threshold = choose_threshold(y.iloc[dev], oof_by_name[winner])
    champion = fit_pipeline(build_pipeline(model, selection, seed), X.iloc[dev], y.iloc[dev])
    # No model/parameter/threshold change below this line may be based on test results.
    scores = champion.predict_proba(X.iloc[test])[:, 1]
    holdout = metrics(y.iloc[test], scores, threshold)
    intervals = bootstrap_intervals(y.iloc[test], scores, threshold, seed)
    dummy_scores = np.full(len(test), y.iloc[dev].mean())
    # Time-aware sensitivity check uses development rows only; no final holdout reused.
    ordered = np.array(sorted(dev, key=lambda idx: (timestamps.iloc[idx], idx)))
    temporal_rows = []
    for fold, (a, b) in enumerate(TimeSeriesSplit(n_splits=3).split(ordered)):
        # Purge identical boundary timestamps to prevent simultaneous runs crossing the boundary.
        training_rows = ordered[a][timestamps.iloc[ordered[a]].to_numpy() < timestamps.iloc[ordered[b[0]]]]
        validation_rows = ordered[b]
        if len(np.unique(y.iloc[training_rows])) != 2 or len(np.unique(y.iloc[validation_rows])) != 2:
            temporal_rows.append({"fold": fold, "status": "skipped: single-class window"})
            continue
        time_model = fit_pipeline(build_pipeline(model, selection, seed), X.iloc[training_rows], y.iloc[training_rows])
        p = time_model.predict_proba(X.iloc[validation_rows])[:, 1]
        temporal_rows.append({"fold": fold, "status": "evaluated", "train_rows": len(training_rows),
                              "validation_rows": len(validation_rows),
                              "train_end": str(timestamps.iloc[training_rows].max()),
                              "validation_start": str(timestamps.iloc[validation_rows].min()),
                              **metrics(y.iloc[validation_rows], p, threshold)})
    quality = {"samples": len(X), "sensors": X.shape[1], "failures": int(y.sum()),
               "passes": int((y == 0).sum()), "failure_rate": float(y.mean()),
               "missing_fraction": float(X.isna().to_numpy().mean()),
               "all_missing_sensors": int(X.isna().all().sum()),
               "constant_sensors": int((X.nunique(dropna=True) <= 1).sum()),
               "timestamp_min": str(timestamps.min()), "timestamp_max": str(timestamps.max()),
               "development_rows": len(dev), "test_rows": len(test), "test_failures": int(y.iloc[test].sum())}
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "matplotlib")}
    result = {"winner": winner, "profile": profile, "seed": seed, "folds": folds,
              "selection_rule": "highest mean fold average precision; lexical tie-break",
              "threshold_rule": "maximize F2 on winner's development out-of-fold predictions",
              "threshold": threshold, "holdout": holdout, "holdout_at_0_5": metrics(y.iloc[test], scores),
              "holdout_intervals_95": intervals, "dummy_holdout": metrics(y.iloc[test], dummy_scores),
              "temporal_sensitivity": temporal_rows, "quality": quality,
              "versions": versions, "python": platform.python_version(),
              "data_sha256": sha256(Path(data_dir) / "secom.data"),
              "labels_sha256": sha256(Path(data_dir) / "secom_labels.data"),
              "probabilities_calibrated": False}
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame({"sample_id": dev, "label": y.iloc[dev].to_numpy(),
                  **oof_by_name}).to_csv(output / "oof_predictions.csv", index=False)
    prediction_table = pd.DataFrame({"sample_id": test, "label": y.iloc[test].to_numpy(),
                                     "failure_score": scores, "prediction": (scores >= threshold).astype(int)})
    prediction_table.to_csv(output / "holdout_predictions.csv", index=False)
    write_json(output / "results.json", result)
    write_json(output / "data_quality.json", quality)
    write_json(output / "missingness.json", X.isna().mean().sort_values(ascending=False).to_dict())
    bundle = {"pipeline": champion, "threshold": threshold, "winner": winner,
              "sensors": SENSORS, "versions": versions, "data_sha256": result["data_sha256"]}
    joblib.dump(bundle, output / "model.joblib")
    X.iloc[test[:5]].to_csv(output / "example_runs.csv", index=False)
    X.iloc[dev[:100]].to_csv(output / "background.csv", index=False)
    # Embedded SQL research store; optional BigQuery exporter is separate.
    with sqlite3.connect(output / "research.sqlite") as database:
        leaderboard.to_sql("experiments", database, if_exists="replace", index=False)
        prediction_table.to_sql("holdout_predictions", database, if_exists="replace", index=False)
    write_json(output / "model_manifest.json", {"sha256": sha256(output / "model.joblib"),
                                               "winner": winner, "versions": versions})
    if mlflow:
        with mlflow.start_run(run_name="selected-model-final-evaluation"):
            mlflow.log_params({"winner": winner, "threshold": threshold, "seed": seed})
            mlflow.log_metrics({k: v for k, v in holdout.items() if isinstance(v, (int, float))})
            for filename in ("model.joblib", "model_manifest.json", "results.json", "leaderboard.csv"):
                mlflow.log_artifact(str(output / filename))
    print(json.dumps({"winner": winner, "holdout": holdout}, indent=2), flush=True)
    return result
