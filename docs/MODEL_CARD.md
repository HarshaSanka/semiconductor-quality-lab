# Model card — SECOM core champion

**Model:** E2_forest_all, RandomForestClassifier, 250 trees, minimum leaf size 3, square-root feature subsampling, balanced subsample class weighting, random seed 42. Training source and versions are recorded in `artifacts/results.json`.

**Intended use:** local educational research, benchmarking, inspection of anonymized predictive signals and demonstration of research engineering methods. A user should reproduce the study before extending it.

**Data:** UCI SECOM, 2008; 1,567 runs, 104 failures, 590 real-valued measurements with missing entries. 1,253 runs used for development, 314 held out. Physical sensor identities, process steps, equipment groups and lot identifiers are unavailable. Data license CC BY 4.0; source DOI 10.24432/C54305.

**Selection:** highest mean development fold AP among five core configurations. Decision threshold 0.08987278061530247 maximizes development OOF F2. Threshold selection and model selection share development data; selected development metrics are optimistic. Holdout was not used to select the model or threshold.

**Inputs:** exactly 590 named numerical sensors, `sensor_000`–`sensor_589`; null/blank allowed for missing entries. All-missing runs, nonnumeric and infinite values, unexpected columns and oversized batches are rejected. A row may still be very incomplete, and a review flag marks missingness above 20%; this heuristic was not validated as an OOD detector.

**Outputs:** uncalibrated failure score, threshold, pass/fail decision, missing fraction and heuristic review flag. Scores near the threshold (absolute difference below 0.05) also trigger review. No per-run statistical uncertainty interval or confidence guarantee is supplied.

**Measured holdout:** AP 0.2008, ROC-AUC 0.7864, precision 0.1416, recall 0.7619, F1 0.2388, Brier 0.05893. 16 true failures detected, 5 missed, 97 false positives, 196 true negatives. See `RESULTS.md` for bootstrap intervals and diagnostic time sensitivity.

**Interpretation:** default global signal ranking is development-fitted impurity importance; it is biased by split opportunities and correlation. Optional SHAP is computed for the served pipeline and labeled with its output units. Neither method reveals physical causation. A PCA champion would use component-level names.

**Out of scope:** autonomous quality release, production process control, setpoint optimization, safety-critical inspection, causal diagnosis, promised yield improvement or direct NGMM/3DHI research claims. High false-alarm rates and temporal instability require further study.

**Reproducibility:** source hashes, split indices, per-fold scores, OOF predictions, holdout predictions, numerical-library versions and a serialized model hash are persisted. The manifest is an integrity check, not an authenticity guarantee. Only load trusted joblib files. Refit after environment changes.

**Further evaluation:** collect recent/group-labeled data, pre-register prospective splits, estimate inspection costs, assess drift and calibration on separate development data, then validate independently. Existing published holdout results cannot guide new tuning and remain an untouched confirmatory test.
