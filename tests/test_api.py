import importlib.util
import unittest

import numpy as np
import pandas as pd

from secom_lab.data import SENSORS
from secom_lab.modeling import build_pipeline, fit_pipeline

AVAILABLE = importlib.util.find_spec("fastapi") is not None and importlib.util.find_spec("httpx") is not None


@unittest.skipUnless(AVAILABLE, "FastAPI/httpx are optional; install .[full,dev]")
class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from secom_lab import api
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(80, 590)), columns=SENSORS)
        model = fit_pipeline(build_pipeline(), X, np.array([0, 1] * 40))
        cls.model = {"pipeline": model, "threshold": 0.5, "winner": "test"}
        api.app.dependency_overrides.clear()
        cls.original = api.bundle
        api.bundle = lambda: cls.model
        cls.api = api
        cls.client = TestClient(api.app)
        cls.row = X.iloc[0].to_dict()

    @classmethod
    def tearDownClass(cls):
        cls.api.bundle = cls.original
        cls.client.close()

    def test_health(self):
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_prediction(self):
        response = self.client.post("/predict", json={"rows": [self.row]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["predictions"]), 1)

    def test_bad_row_schema(self):
        self.assertEqual(self.client.post("/predict", json={"rows": [{"sensor_000": 1}]}).status_code, 422)

    def test_empty_batch(self):
        self.assertEqual(self.client.post("/predict", json={"rows": []}).status_code, 422)

    def test_all_null_row(self):
        self.assertEqual(self.client.post("/predict", json={"rows": [dict.fromkeys(SENSORS)]}).status_code, 422)

    def test_explanation_batch_limit(self):
        self.assertEqual(self.client.post("/explain", json={"rows": [self.row] * 11}).status_code, 422)
