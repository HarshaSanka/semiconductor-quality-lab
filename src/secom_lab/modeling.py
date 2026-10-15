"""All learned preprocessing lives inside each cross-validation pipeline."""
from __future__ import annotations

from functools import partial

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MissingnessFilter(TransformerMixin, BaseEstimator):
    def __init__(self, max_missing=0.5):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.n_features_in_ = values.shape[1]
        self.support_ = np.isnan(values).mean(axis=0) <= self.max_missing
        if not self.support_.any():
            raise ValueError("No eligible sensors remain")
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Sensor dimension changed")
        return values[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features if input_features is not None else
                           [f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return names[self.support_]


def experiments(profile="core"):
    configs = {
        "E1_logistic_all": ("logistic", "all"),
        "E2_forest_all": ("forest", "all"),
        "E3_forest_mi40": ("forest", "mi"),
        "E4_forest_rfe40": ("forest", "rfe"),
        "E5_logistic_pca40": ("logistic", "pca"),
    }
    if profile == "full":
        import xgboost  # fail early rather than silently substituting another algorithm
        assert xgboost
        configs.update({"E6_xgboost_all": ("xgboost", "all"),
                        "E7_xgboost_rfe40": ("xgboost", "rfe"),
                        "E8_xgboost_pca40": ("xgboost", "pca")})
    return configs


def build_pipeline(model="logistic", selection="all", seed=42):
    steps = [("missingness", MissingnessFilter()), ("imputer", SimpleImputer(strategy="median")),
             ("variance", VarianceThreshold()), ("scale", StandardScaler())]
    if selection == "mi":
        steps.append(("select", SelectKBest(partial(mutual_info_classif, random_state=seed), k=40)))
    elif selection == "rfe":
        steps.append(("select", RFE(LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear",
                                                       random_state=seed, max_iter=2000),
                                      n_features_to_select=40, step=0.25)))
    elif selection == "pca":
        steps.append(("select", PCA(n_components=40, svd_solver="full")))
    elif selection != "all":
        raise ValueError(f"Unknown selection: {selection}")
    if model == "logistic":
        classifier = LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear",
                                        max_iter=2000, random_state=seed)
    elif model == "forest":
        classifier = RandomForestClassifier(n_estimators=250, min_samples_leaf=3, max_features="sqrt",
                                            class_weight="balanced_subsample", n_jobs=1, random_state=seed)
    elif model == "xgboost":
        from xgboost import XGBClassifier
        classifier = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.04, subsample=0.8,
                                   colsample_bytree=0.8, reg_lambda=5, tree_method="hist",
                                   n_jobs=1, random_state=seed, eval_metric="logloss")
    else:
        raise ValueError(f"Unknown model: {model}")
    return Pipeline([*steps, ("model", classifier)])


def fit_pipeline(pipeline, X, y):
    kwargs = {}
    if pipeline[-1].__class__.__module__.startswith("xgboost"):
        from sklearn.utils.class_weight import compute_sample_weight
        kwargs["model__sample_weight"] = compute_sample_weight("balanced", y)
    return pipeline.fit(X, y, **kwargs)
