"""Shared inference logic for CLI, API and dashboard."""
from pathlib import Path
import json

import joblib
import numpy as np

from .data import sha256, validate_frame


def load_bundle(directory=Path("artifacts")):
    directory = Path(directory)
    manifest = json.loads((directory / "model_manifest.json").read_text())
    if sha256(directory / "model.joblib") != manifest["sha256"]:
        raise ValueError("Model artifact hash mismatch")
    # Joblib is executable serialization. Load only your own trusted artifact directories.
    return joblib.load(directory / "model.joblib")


def predict(bundle, frame):
    frame = validate_frame(frame)
    scores = bundle["pipeline"].predict_proba(frame)[:, 1]
    missing = frame.isna().mean(axis=1).to_numpy()
    return [{"row": i, "failure_score": float(score),
             "prediction": "fail" if score >= bundle["threshold"] else "pass",
             "threshold": float(bundle["threshold"]), "missing_fraction": float(missing[i]),
             "review_required": bool(missing[i] > 0.2 or abs(score - bundle["threshold"]) < 0.05),
             "note": "Uncalibrated model score; review flag is a heuristic, not an uncertainty interval."}
            for i, score in enumerate(scores)]


def explain(bundle, frame, background):
    import shap
    pipeline = bundle["pipeline"]
    frame = validate_frame(frame)
    transformed = pipeline[:-1].transform(frame)
    base = pipeline[:-1].transform(validate_frame(background))
    names = pipeline[:-1].get_feature_names_out(bundle["sensors"])
    model = pipeline[-1]
    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, base)
        explanation = explainer(transformed)
        units = "log-odds"
    else:
        explainer = shap.TreeExplainer(model, data=base, feature_perturbation="interventional",
                                       model_output="probability")
        explanation = explainer(transformed, check_additivity=False)
        units = "failure-score contribution"
    values, bases = explanation.values, explanation.base_values
    if values.ndim == 3:
        values, bases = values[:, :, 1], bases[:, 1]
    return {"features": list(names), "values": np.asarray(values).tolist(),
            "base_values": np.asarray(bases).tolist(), "units": units,
            "note": "Attributions explain the model; they are not physical causes. PCA names are components."}
