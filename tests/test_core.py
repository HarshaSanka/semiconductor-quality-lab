"""Meaningful contracts: leakage, schema, serialization and metric behavior."""
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from secom_lab.data import SENSORS, read_csv, sha256, validate_frame
from secom_lab.evaluation import bootstrap_intervals, choose_threshold, metrics
from secom_lab.inference import load_bundle, predict
from secom_lab.modeling import MissingnessFilter, build_pipeline, fit_pipeline


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(np.ones((2, 590)), columns=SENSORS)

    def test_reordered_columns_are_canonicalized(self):
        self.assertEqual(list(validate_frame(self.frame.iloc[:, ::-1])), SENSORS)

    def test_missing_columns_rejected(self):
        with self.assertRaises(ValueError):
            validate_frame(self.frame.drop(columns=SENSORS[0]))

    def test_label_leakage_column_rejected(self):
        self.frame["label"] = 1
        with self.assertRaises(ValueError):
            validate_frame(self.frame)

    def test_infinity_rejected(self):
        self.frame.iloc[0, 0] = np.inf
        with self.assertRaises(ValueError):
            validate_frame(self.frame)

    def test_missing_values_allowed(self):
        self.frame.iloc[0, 0] = np.nan
        self.assertTrue(np.isnan(validate_frame(self.frame).iloc[0, 0]))

    def test_all_missing_run_rejected(self):
        self.frame.iloc[0, :] = np.nan
        with self.assertRaises(ValueError):
            validate_frame(self.frame)

    def test_empty_batch_rejected(self):
        with self.assertRaises(ValueError):
            validate_frame(self.frame.iloc[:0])

    def test_oversized_batch_rejected(self):
        with self.assertRaises(ValueError):
            validate_frame(pd.concat([self.frame] * 501))

    def test_non_numeric_rejected(self):
        self.frame = self.frame.astype(object)
        self.frame.iloc[0, 0] = "bad"
        with self.assertRaises(ValueError):
            validate_frame(self.frame)

    def test_csv_duplicate_header_rejected(self):
        content = self.frame.to_csv(index=False).replace("sensor_001", "sensor_000", 1).encode()
        with self.assertRaises(ValueError):
            read_csv(content)

    def test_csv_roundtrip(self):
        pd.testing.assert_frame_equal(read_csv(self.frame.to_csv(index=False).encode()), self.frame)


class ModelingTests(unittest.TestCase):
    def test_missingness_filter_never_refits_on_validation(self):
        train = np.array([[1, np.nan, 4], [2, np.nan, 5], [3, 9, 6]], dtype=float)
        filter_ = MissingnessFilter().fit(train)
        before = filter_.support_.copy()
        validation = np.array([[np.nan, 2, 8]], dtype=float)
        self.assertEqual(filter_.transform(validation).shape[1], 2)
        np.testing.assert_array_equal(before, filter_.support_)

    def test_imputer_uses_only_training_medians(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(60, 8))
        X[0, 0] = np.nan
        pipeline = fit_pipeline(build_pipeline(), X, np.array([0, 1] * 30))
        expected = np.nanmedian(X, axis=0)
        np.testing.assert_allclose(pipeline["imputer"].statistics_, expected)
        pipeline.predict_proba(np.full((1, 8), 10000))
        np.testing.assert_allclose(pipeline["imputer"].statistics_, expected)

    def test_preprocessing_and_model_serialization(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(60, 590)), columns=SENSORS)
        X.iloc[0, 0] = np.nan
        model = fit_pipeline(build_pipeline(), X, np.array([0, 1] * 30))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump({"pipeline": model, "threshold": 0.3, "winner": "unit-test"}, path)
            (Path(folder) / "model_manifest.json").write_text(json.dumps({"sha256": sha256(path)}))
            loaded = load_bundle(Path(folder))
            np.testing.assert_allclose(model.predict_proba(X[:2]), loaded["pipeline"].predict_proba(X[:2]))
            self.assertEqual(len(predict(loaded, X[:2])), 2)

    def test_hash_mismatch_prevents_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "model.joblib").write_bytes(b"invalid")
            (path / "model_manifest.json").write_text('{"sha256":"wrong"}')
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_bundle(path)

    def test_feature_names_survive_sensor_selection(self):
        rng = np.random.default_rng(2)
        X = pd.DataFrame(rng.normal(size=(100, 590)), columns=SENSORS)
        pipeline = fit_pipeline(build_pipeline(selection="rfe"), X, np.array([0, 1] * 50))
        names = pipeline[:-1].get_feature_names_out(SENSORS)
        self.assertEqual(len(names), 40)
        self.assertTrue(set(names) <= set(SENSORS))


class EvaluationTests(unittest.TestCase):
    def test_failure_is_positive_class(self):
        report = metrics([0, 0, 1, 1], [0.1, 0.8, 0.9, 0.2], threshold=0.5)
        self.assertEqual([report[k] for k in ("tn", "fp", "fn", "tp")], [1, 1, 1, 1])
        self.assertEqual(report["recall"], 0.5)

    def test_ap_is_not_accuracy(self):
        y, score = [0] * 90 + [1] * 10, [0.1] * 100
        self.assertAlmostEqual(metrics(y, score)["average_precision"], 0.1)

    def test_threshold_is_f2_optimal_on_development(self):
        y, score = np.array([0, 1, 0, 1, 0]), np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        threshold = choose_threshold(y, score)
        def f2(t):
            m = metrics(y, score, t)
            return 5 * m["precision"] * m["recall"] / max(4 * m["precision"] + m["recall"], 1e-12)
        self.assertAlmostEqual(f2(threshold), max(f2(t) for t in score))

    def test_bootstrap_reproducible_and_ordered(self):
        y, scores = [0, 0, 0, 1, 1], [0.1, 0.2, 0.6, 0.5, 0.8]
        a = bootstrap_intervals(y, scores, 0.4, repeats=20)
        self.assertEqual(a, bootstrap_intervals(y, scores, 0.4, repeats=20))
        self.assertTrue(all(low <= high for low, high in a.values()))


class ArtifactTests(unittest.TestCase):
    def test_published_splits_and_metrics(self):
        root = Path("artifacts")
        if not (root / "split_manifest.json").exists():
            self.skipTest("Run training to validate experiment artifacts")
        manifest = json.loads((root / "split_manifest.json").read_text())
        dev, test = set(manifest["development"]), set(manifest["test"])
        self.assertFalse(dev & test)
        self.assertEqual(len(dev | test), 1567)
        validation_seen = []
        for fold in manifest["folds"]:
            train, validation = set(fold["train"]), set(fold["validation"])
            self.assertFalse(train & validation)
            self.assertEqual(train | validation, dev)
            validation_seen.extend(validation)
        self.assertEqual(len(validation_seen), len(set(validation_seen)))
        predictions = pd.read_csv(root / "holdout_predictions.csv")
        self.assertEqual(set(predictions.sample_id), test)
        results = json.loads((root / "results.json").read_text())
        self.assertAlmostEqual(average_precision_score(predictions.label, predictions.failure_score),
                               results["holdout"]["average_precision"])
        for window in results["temporal_sensitivity"]:
            if window["status"] == "evaluated":
                self.assertLess(window["train_end"], window["validation_start"])


if __name__ == "__main__":
    unittest.main()
