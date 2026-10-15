"""Optional explicit append of experiment metrics to an existing BigQuery dataset."""
import argparse
import uuid
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True, help="project.dataset.table (existing dataset)")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--execute", action="store_true", help="Actually submit a load job; default is local preview")
    args = parser.parse_args()
    frame = pd.read_csv(args.artifacts / "leaderboard.csv")
    frame["benchmark_run_id"] = str(uuid.uuid4())
    frame["exported_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    print(frame.to_json(orient="records", indent=2))
    if args.execute:
        from google.cloud import bigquery
        client = bigquery.Client()
        job = client.load_table_from_dataframe(frame, args.table,
                    job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"))
        job.result()
        print(f"Appended {len(frame)} experiment rows to {args.table}")


if __name__ == "__main__":
    main()
