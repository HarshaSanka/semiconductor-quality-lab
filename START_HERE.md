# Your project is ready to inspect

Read **README.md** for setup and measured results. Open **docs/results_viewer.html** in your browser for the offline interactive result pages. Read **docs/SHOWCASE.md** for the demonstration script, resume bullets and interview questions.

The package includes the real public SECOM data, fitted core model, research outputs and source. A Python 3.12 environment with the numerical dependencies is still required to execute predictions. Dependencies are not bundled in this archive.

Core verification: **25 tests passed, six optional API tests skipped**, plus real-data training, CLI inference, SQL queries and syntax checks. Optional XGBoost, SHAP, MLflow, FastAPI, Streamlit, Docker and BigQuery integrations are supplied but were not runtime-verified in the restricted build environment.

## Publish the code

The public repository is https://github.com/HarshaSanka/semiconductor-quality-lab. Use the README and measured research report as the main project entry points. The commands below are for publishing your own subsequent changes.

From this folder, after installing GitHub CLI if necessary:

```bash
gh auth login --hostname github.com
bash scripts/publish_github.sh
```

The publisher uses the existing repository and never force-pushes. Git may ask you to configure your own commit name/email if none is configured. The source repository ignores raw data, executable models and local environments; collaborators can regenerate artifacts using the README. Do not upload the entire archive as a single source file—publish the extracted project tree.

After publishing, pin the repository to your profile and use its URL on your resume. Present the measured false-alarm trade-off alongside recall. Do not claim production optimization or direct 3DHI results.
