"""Failure-focused metrics and development-only decision threshold selection."""
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)


def metrics(y, scores, threshold=0.5):
    prediction = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {"average_precision": float(average_precision_score(y, scores)),
            "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)),
            "f1": float(f1_score(y, prediction, zero_division=0)),
            "brier": float(brier_score_loss(y, scores)), "threshold": float(threshold),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def choose_threshold(y, oof_scores):
    precision, recall, thresholds = precision_recall_curve(y, oof_scores)
    f2 = 5 * precision[:-1] * recall[:-1] / np.maximum(4 * precision[:-1] + recall[:-1], 1e-12)
    # First maximum: deterministic tie-break toward the lower threshold/higher recall.
    return float(thresholds[np.argmax(f2)])


def bootstrap_intervals(y, scores, threshold, seed=42, repeats=1000):
    """Stratified percentile intervals conditional on observed class counts."""
    rng = np.random.default_rng(seed)
    y, scores = np.asarray(y), np.asarray(scores)
    groups = [np.flatnonzero(y == label) for label in (0, 1)]
    if any(not len(group) for group in groups):
        return {}
    estimates = []
    for _ in range(repeats):
        rows = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        estimates.append(metrics(y[rows], scores[rows], threshold))
    return {key: [float(np.quantile([m[key] for m in estimates], q)) for q in (0.025, 0.975)]
            for key in ("average_precision", "roc_auc", "precision", "recall", "f1", "brier")}
