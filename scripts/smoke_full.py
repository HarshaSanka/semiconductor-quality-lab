"""Optional-stack smoke check: XGBoost, SHAP additivity and MLflow artifact tracking."""
import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from scipy.special import expit

from secom_lab.data import SENSORS
from secom_lab.inference import explain
from secom_lab.modeling import build_pipeline, fit_pipeline

rng = np.random.default_rng(42)
X = pd.DataFrame(rng.normal(size=(120, 590)), columns=SENSORS)
y = np.array([0] * 100 + [1] * 20)
X.loc[y == 1, "sensor_000"] += 2
X.iloc[0, 10] = np.nan
with tempfile.TemporaryDirectory() as folder:
    mlflow.set_tracking_uri("sqlite:///" + str(Path(folder) / "tracking.db"))
    mlflow.set_experiment("secom-integration-smoke")
    with mlflow.start_run():
        for kind in ("logistic", "forest", "xgboost"):
            model = fit_pipeline(build_pipeline(kind), X, y)
            bundle = {"pipeline": model, "sensors": SENSORS}
            explanation = explain(bundle, X.iloc[-2:], X.iloc[:40])
            total = np.array(explanation["base_values"]) + np.array(explanation["values"]).sum(axis=1)
            if explanation["units"] == "log-odds":
                total = expit(total)
            np.testing.assert_allclose(total, model.predict_proba(X.iloc[-2:])[:, 1], atol=1e-5)
            mlflow.log_metric(kind + "_score", float(total.mean()))
        output = Path(folder) / "smoke.json"
        output.write_text(json.dumps({"status": "passed", "models": ["logistic", "forest", "xgboost"]}))
        mlflow.log_artifact(str(output))
print("XGBoost / SHAP additivity / MLflow smoke checks passed")
