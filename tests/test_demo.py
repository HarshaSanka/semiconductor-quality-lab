"""Exercise HTTP handlers without opening a listening socket."""
import io
import json
import unittest
from pathlib import Path

from secom_lab.demo import create_handler


@unittest.skipUnless(Path("artifacts/model.joblib").exists(), "Train core artifacts first")
class DemoHandlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.handler_type = create_handler(Path("artifacts"))

    def handler(self, path, content=b""):
        handler = object.__new__(self.handler_type)
        handler.path = path
        handler.headers = {"Content-Length": str(len(content))}
        handler.rfile = io.BytesIO(content)
        self.responses = []
        handler.send = lambda *args: self.responses.append(args)
        return handler

    def test_serves_results(self):
        self.handler("/api/results").do_GET()
        self.assertEqual(self.responses[0][0], 200)
        self.assertIn("holdout", json.loads(self.responses[0][1]))

    def test_predicts_csv(self):
        self.handler("/api/predict", Path("artifacts/example_runs.csv").read_bytes()).do_POST()
        self.assertEqual(self.responses[0][0], 200)
        self.assertEqual(len(json.loads(self.responses[0][1])), 5)

    def test_bad_csv_returns_validation_error(self):
        self.handler("/api/predict", b"wrong\n1\n").do_POST()
        self.assertEqual(self.responses[0][0], 422)

    def test_unknown_path_rejected(self):
        self.handler("/../../secret").do_GET()
        self.assertEqual(self.responses[0][0], 404)
