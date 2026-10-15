"""Build portfolio figures, a measured research report and an offline results viewer."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from secom_lab.inference import load_bundle

root = Path("artifacts")
assets = Path("docs/assets")
assets.mkdir(parents=True, exist_ok=True)
r = json.loads((root / "results.json").read_text())
board = pd.read_csv(root / "leaderboard.csv")
predictions = pd.read_csv(root / "holdout_predictions.csv")
m, q = r["holdout"], r["quality"]
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
labels = board.experiment.str.replace("_", " ")
axes[0].barh(labels[::-1], board.cv_ap_mean.iloc[::-1], xerr=board.cv_ap_std.iloc[::-1], color="#147d92", capsize=3)
axes[0].set(xlabel="Mean fold average precision ± fold standard deviation", title="Development benchmark · 5 stratified folds")
precision, recall, _ = precision_recall_curve(predictions.label, predictions.failure_score)
axes[1].step(recall, precision, where="post", color="#147d92", label=f'Champion AP = {m["average_precision"]:.3f}')
axes[1].axhline(predictions.label.mean(), linestyle="--", color="#a4651a", label="Constant-score baseline")
axes[1].scatter([m["recall"]], [m["precision"]], color="#d35a41", zorder=4, label="Development-selected threshold")
axes[1].set(xlabel="Failure recall", ylabel="Failure precision", title="Final holdout · 314 runs / 21 failures", xlim=(0, 1), ylim=(0, 1.02))
axes[1].legend(fontsize=8)
fig.savefig(assets / "benchmark.png", facecolor="white")
plt.close(fig)
fig, ax = plt.subplots(figsize=(5.4, 4.5), layout="constrained")
matrix = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
ax.imshow(matrix, cmap="Blues")
for (row, col), count in np.ndenumerate(matrix):
    ax.text(col, row, str(count), ha="center", va="center", fontsize=28,
            color="white" if count > matrix.max() / 2 else "#112d45")
ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Predicted pass", "Predicted fail"],
       yticklabels=["Actual pass", "Actual fail"], title=f'Holdout confusion matrix · threshold {r["threshold"]:.4f}')
fig.savefig(assets / "confusion_matrix.png", facecolor="white")
plt.close(fig)
bundle = load_bundle(root)
pipeline = bundle["pipeline"]
names = pipeline[:-1].get_feature_names_out(bundle["sensors"])
model = pipeline[-1]
importance = getattr(model, "feature_importances_", np.abs(getattr(model, "coef_", np.zeros((1, len(names))))[0]))
features = pd.DataFrame({"feature": names, "importance": importance}).sort_values("importance", ascending=False)
features.to_csv(root / "feature_importance.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
top = features.head(15).iloc[::-1]
ax.barh(top.feature, top.importance, color="#147d92")
ax.set(xlabel="Training-fitted model importance (impurity decrease for Random Forest)",
       title="Predictive sensor signals · exploratory, not causal")
fig.savefig(assets / "sensor_importance.png", facecolor="white")
plt.close(fig)

table = "| Experiment | CV AP mean | Fold SD |\n| --- | ---: | ---: |\n" + "\n".join(
    f"| {row.experiment} | {row.cv_ap_mean:.4f} | {row.cv_ap_std:.4f} |" for row in board.itertuples())
metric_table = "| Metric | Measured value | 95% interval |\n| --- | ---: | --- |\n" + "\n".join(
    f'| {key} | {m[key]:.4f} | {r["holdout_intervals_95"][key][0]:.4f}–{r["holdout_intervals_95"][key][1]:.4f} |'
    for key in ("average_precision", "roc_auc", "precision", "recall", "f1", "brier"))
report = f"""# Measured research results

This is a small-data predictive study, not a manufacturing optimization system. The core benchmark was executed on real SECOM measurements. The full optional stack was not executed in the build environment.

## Dataset and protocol

{q['samples']} runs, {q['sensors']} anonymized measurements, {q['failures']} failures and {q['passes']} passes. Missing measurements account for {q['missing_fraction']:.2%} of cells; {q['constant_sensors']} columns have at most one observed value. Sensor numbering is zero-based. Original labels −1 / +1 map to pass=0 / fail=1. UCI metadata reports 591 features; the verified raw matrix contains 590 measurement columns.

Use a seeded 80/20 stratified split: {q['development_rows']} development rows and {q['test_rows']} holdout rows, including {q['test_failures']} failures. Every experiment uses the same five stratified development folds. Each fold learns its >50% missingness filter, median imputer, variance filter, scaler, selector and classifier using training rows only. Outliers are retained: without sensor semantics, removing large measurements could remove failure signals. Scaling and tree comparisons probe sensitivity; no physical outlier limits are invented.

Rank by mean fold **average precision (AP)**. AP is a non-interpolated precision–recall summary, not trapezoidal PR-AUC. Choose the winner before inspecting the holdout. Use that winner's out-of-fold development scores to maximize F2, weighting recall more than precision. These development selection statistics are optimistic after model/threshold selection; they are not nested-CV estimates.

## Development benchmark

{table}

The selected model is **{r['winner']}**, with threshold **{r['threshold']:.6f}**. RFE uses a class-balanced logistic estimator and selects 40 variables; the final RFE experiment uses a forest. MI selects 40 variables. PCA retains 40 components. Class weights address imbalance; XGBoost's optional path uses per-training-fold balanced sample weights. The benchmark compares configurations, not a factorial causal study of each component.

![Model comparison and precision–recall curve](assets/benchmark.png)

## Final holdout evaluation

{metric_table}

The model detects **{m['tp']} of {m['tp']+m['fn']} failures**, misses **{m['fn']}**, and flags **{m['fp']} passing runs** as failures. It correctly passes {m['tn']} runs. Failure precision is only {m['precision']:.1%}; the false-positive rate is {m['fp']/(m['fp']+m['tn']):.1%}. This is a substantial inspection burden. Default threshold 0.5 detects no failures; the development-selected threshold materially changes the operating point.

The constant-score baseline has AP {r['dummy_holdout']['average_precision']:.4f} and ROC-AUC 0.5000. Ranking performance exceeds that baseline on this holdout, but this result does not establish industrial usefulness. Scores are uncalibrated; Brier score is reported as a diagnostic, not proof of calibration. No per-run uncertainty interval is supplied.

Intervals use 1,000 stratified bootstrap resamples of holdout predictions with seed 42. They condition on observed class counts, do not repeat model selection or fitting, and assume independent runs. They understate uncertainty when runs are correlated and are especially fragile with only 21 failures.

![Confusion matrix](assets/confusion_matrix.png)

## Chronological sensitivity

Sort development rows by timestamp and use three expanding windows. Purge training rows sharing the validation boundary timestamp. Window AP values are {', '.join(format(w['average_precision'], '.4f') for w in r['temporal_sensitivity'] if w['status']=='evaluated')}. The earliest window raises many false alarms; later performance also varies. These checks demonstrate sensitivity to time and training-window size.

The model configuration and threshold were selected using all development rows, including later windows. Therefore these are descriptive chronological sensitivity checks, **not an independent prospective forecast evaluation**. The primary random holdout can also mix correlated production conditions across splits. No lot/equipment group identifiers are available to implement group holdouts.

## Interpretation and limitations

![Model sensor importance](assets/sensor_importance.png)

These are model importances learned on development data. Impurity importance can favor variables with many possible splits and distribute importance unpredictably among correlated sensors. It does not identify physical mechanisms. Optional SHAP explanations label their units and explain the actual served model; they do not establish causation. PCA explanations refer to latent components, not named physical sensors.

This historical 2008 dataset has anonymized measurements, few positive labels, unknown production grouping and possible drift. No causal process optimization, yield improvement, 3DHI result, DARPA endorsement or production deployment is claimed. Future work should pre-register a chronological/grouped evaluation, collect more failures and process context, calibrate scores with separate development data, evaluate inspection costs and validate prospectively.

## Reproducibility and execution status

Dataset SHA-256: `{r['data_sha256']}`. Label SHA-256: `{r['labels_sha256']}`. Split indices, fold metrics, OOF scores, holdout scores, model manifest and exact observed core versions are saved under `artifacts/`.

Executed: real-data core training, five configurations × five folds, final holdout, 1,000 bootstrap resamples, three time sensitivity checks, SQL storage, CLI predictions and core contract tests. Optional API tests are skipped when FastAPI/httpx are absent. XGBoost, SHAP, MLflow, Streamlit, Docker and BigQuery integrations are included but **not runtime-verified here**. Network dependency installation and localhost binding were unavailable. CI configuration is supplied; no remote CI success is claimed.

## Source

McCann, M., & Johnston, A. (2008). *SECOM* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54305. Dataset license: CC BY 4.0. The local archive was structurally checked and hashed; UCI does not supply an official checksum used by this project.
"""
Path("docs/RESULTS.md").write_text(report)
# Build a file://-compatible results viewer. Live predictions remain available in the local server.
html = (Path("src/secom_lab/web.html")).read_text()
payloads = {"/api/results": r, "/api/missingness": json.loads((root / "missingness.json").read_text()),
            "/api/leaderboard": (root / "leaderboard.csv").read_text()}
embed = "<script>const SNAPSHOT=" + json.dumps(payloads).replace("<", "\\u003c") + ";window.fetch=async url=>({json:async()=>SNAPSHOT[url],text:async()=>SNAPSHOT[url]});</script>"
html = html.replace("<script>\nconst $", embed + "<script>\nconst $")
html = html.replace('<section id="prediction" class="hidden">', '<section id="prediction" class="hidden"><div class="note">Offline results snapshot. For working CSV predictions run <code>python -m secom_lab.demo</code> and open its localhost URL.</div>')
html = html.replace('<button id="predict" class="primary">', '<button id="predict" class="primary" disabled>')
html = html.replace('href="/example.csv"', 'href="../artifacts/example_runs.csv"')
Path("docs/results_viewer.html").write_text(html)
Path("requirements-core-observed.txt").write_text("# Exact versions used for the included experiment; Python " + r["python"] + "\n" +
    "\n".join(f"{name}=={version}" for name, version in r["versions"].items()) + "\n")
print("Built figures, docs/RESULTS.md and docs/results_viewer.html")
