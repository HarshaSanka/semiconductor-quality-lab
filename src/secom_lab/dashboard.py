"""Optional Streamlit research dashboard; inference goes through FastAPI."""
import json
import os
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

from secom_lab.data import read_csv

root = Path(os.environ.get("SECOM_ARTIFACTS", "artifacts"))
api_url = os.environ.get("SECOM_API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="Semiconductor Quality Lab", page_icon="◈", layout="wide")
st.title("Semiconductor Quality Lab")
st.caption("SECOM · Research prototype · Failure is the positive class")
if not (root / "results.json").exists():
    st.info("Run secom-lab download, then secom-lab train first.")
    st.stop()
results = json.loads((root / "results.json").read_text())
page = st.sidebar.radio("Research workspace", ["Data overview", "Quality prediction", "Model comparison", "Explainable AI"])
if page == "Data overview":
    quality = results["quality"]
    for column, (label, value) in zip(st.columns(4), [("Runs", quality["samples"]), ("Sensors", quality["sensors"]),
                                                     ("Failures", quality["failures"]), ("Failure rate", f'{quality["failure_rate"]:.1%}')]):
        column.metric(label, value)
    missing = pd.Series(json.loads((root / "missingness.json").read_text()), name="missing_fraction")
    st.plotly_chart(px.bar(missing.head(25), title="Sensors with the most missing measurements"), use_container_width=True)
    st.write("Sensor names refer to zero-based raw measurement columns. Their physical meanings are unavailable.")
elif page == "Model comparison":
    st.dataframe(pd.read_csv(root / "leaderboard.csv"), hide_index=True, use_container_width=True)
    st.write("Models are ranked using mean development CV average precision. Only the selected model receives final test evaluation.")
    st.json({"winner": results["winner"], "holdout": results["holdout"], "intervals_95": results["holdout_intervals_95"]})
else:
    st.download_button("Download five example holdout runs", (root / "example_runs.csv").read_bytes(), "example_runs.csv", "text/csv")
    upload = st.file_uploader("Upload a CSV with all 590 named sensors", type="csv")
    if upload is not None:
        try:
            frame = read_csv(upload.getvalue())
            rows = json.loads(frame.to_json(orient="records"))
            endpoint = "/predict" if page == "Quality prediction" else "/explain"
            if st.button("Analyze runs"):
                response = httpx.post(api_url + endpoint, json={"rows": rows}, timeout=120)
                response.raise_for_status()
                payload = response.json()
                if endpoint == "/predict":
                    st.dataframe(payload["predictions"], hide_index=True)
                    st.download_button("Download results", json.dumps(payload, indent=2), "predictions.json")
                else:
                    feature = pd.DataFrame({"feature": payload["features"], "contribution": payload["values"][0]})
                    feature = feature.loc[feature.contribution.abs().nlargest(15).index]
                    st.plotly_chart(px.bar(feature, x="contribution", y="feature", orientation="h",
                                           title=f'First run · {payload["units"]}'))
                    st.caption(payload["note"])
        except (ValueError, httpx.HTTPError) as error:
            st.error(str(error))
st.warning("Scores are uncalibrated. This prototype supports investigation; it does not establish physical causes or recommend process settings.")
