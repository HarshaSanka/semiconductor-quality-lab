# GitHub and interview showcase

## Repository presentation

Suggested name: **semiconductor-quality-lab**

Description: **Reproducible semiconductor failure-prediction research on UCI SECOM, with measured benchmarks, leakage-aware pipelines, strict inference contracts and an inspection dashboard.**

Topics: `semiconductor`, `machine-learning`, `quality-prediction`, `scikit-learn`, `research-engineering`, `explainable-ai`, `fastapi`, `streamlit`, `reproducibility`, `secom`.

Lead with the measured results and their inspection trade-off. Keep the README benchmark figure visible. Pin the repository to your profile and link directly to `docs/RESULTS.md` when sharing technical evidence. Use `docs/results_viewer.html` for an offline demonstration. GitHub shows its source; download/open it to interact.

## Publish from your computer

Repository: https://github.com/HarshaSanka/semiconductor-quality-lab. To publish subsequent local changes from the project root:

```bash
gh auth login --hostname github.com
bash scripts/publish_github.sh
```

For a separate future repository, the generic creation sequence is:

```bash
gh auth login --hostname github.com
git init -b main
git add .
git commit -m "Add SECOM research platform and measured core benchmark"
gh repo create semiconductor-quality-lab --public --source=. --remote=origin --push
gh repo edit --description "Reproducible semiconductor failure prediction on UCI SECOM with measured benchmarks and research tooling"
gh repo edit --add-topic semiconductor --add-topic machine-learning --add-topic secom --add-topic research-engineering
```

Only initialize/create if needed. If a remote already exists, use its normal commit/push flow. The source repo ignores data and executable models; collaborators reproduce them with the README commands. Do not add secrets, local environments or `mlruns` to Git. CI is configured but a workflow file alone is not evidence of a successful run.

## Two-minute demonstration

1. Open the README and explain the research question: can sensor patterns help flag failed manufacturing runs?
2. Open the benchmark figure. Explain the fixed development folds and why average precision matters when only 6.6% of runs fail.
3. Show the confusion matrix: 16 failures detected, five missed, 97 false alarms. Describe the review burden honestly.
4. Open the results viewer. Compare model configurations, then explain that only the CV champion received final evaluation.
5. Run `secom-lab predict artifacts/example_runs.csv`. Explain strict schema checks and training-fitted preprocessing. For a live browser demo, run `python -m secom_lab.demo` outside a restricted environment.
6. End with one limitation: anonymous sensors prevent physical diagnosis, and historical random-split results need prospective validation.

## Resume wording supported by the executed core

**Semiconductor Quality Prediction & Research Analytics — Personal Project**

- Built reproducible Python/scikit-learn pipelines to benchmark five model and feature-selection configurations on 1,567 SECOM manufacturing runs with 590 anonymized measurements.
- Implemented training-only preprocessing, failure-focused cross-validation, development-based threshold selection, holdout evaluation and stratified bootstrap intervals; the selected model detected 16 of 21 holdout failures with 97 false alarms.
- Developed strict batch-inference contracts, saved artifact provenance, SQLite research outputs and a browser-based results viewer, supported by automated core contract tests.

Do not say XGBoost, SHAP, MLflow, FastAPI, Streamlit, Docker or BigQuery were operationally validated until you install, execute and inspect those integrations. The repository includes their code; that is different from having measured operational evidence. Do not claim sole manual authorship of an AI-assisted implementation.

## Natural interview explanation

“I wanted a project that connects my electronics background with AI research, so I studied failure prediction using the public SECOM manufacturing dataset. It has many anonymized measurements, missing values and relatively few failures. I compared several modeling and feature-selection approaches using the same development folds and kept preprocessing inside those folds. I chose the model and decision threshold before evaluating the holdout. The Random Forest caught 16 of 21 failures, but it also raised 97 false alarms, so I would describe it as a research prototype that needs more validation. I built inference tools and a results viewer so the findings can be inspected and reproduced. The important next step would be recent data with process and lot context for a prospective evaluation.”

Use this only after reproducing the project and understanding each part. Adapt the phrasing to your own voice and contributions.

## Questions to prepare

| Question | Evidence to discuss |
| --- | --- |
| Why not accuracy? | A model predicting every run passes is about 93% accurate but misses every failure. |
| How did you avoid leakage? | Fitted preprocessing/selection inside each training fold; final holdout excluded from selection. |
| Why a threshold below 0.5? | Balanced models produce uncalibrated scores; OOF F2 favors recall. Discuss false alarms. |
| Is AP the same as trapezoidal PR-AUC? | No. The project explicitly uses sklearn average precision. |
| Does feature importance identify the failed process? | No. Sensors are anonymous, correlation is present and attribution is model-dependent. |
| Is the time check prospective? | No. Configuration and threshold used all development data; windows are diagnostic sensitivity checks. |
| What did the AI assistant do? | Helped implement and document the initial version; explain your own reproduction, review, modifications and conclusions. |
| What would production require? | Fresh/grouped validation, calibration, cost thresholds, drift checks and integration with known process context. |
