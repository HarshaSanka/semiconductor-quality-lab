import argparse
import json
from pathlib import Path

from .data import acquire, read_csv


def main():
    parser = argparse.ArgumentParser(description="SECOM research lab")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download")
    download.add_argument("--archive", type=Path)
    download.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    train = sub.add_parser("train")
    train.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    train.add_argument("--output", type=Path, default=Path("artifacts"))
    train.add_argument("--profile", choices=["core", "full"], default="core")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--tracking-uri")
    prediction = sub.add_parser("predict")
    prediction.add_argument("csv", type=Path)
    prediction.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.command == "download":
        print(json.dumps(acquire(args.data_dir, args.archive), indent=2))
    elif args.command == "train":
        from .train import run
        run(args.data_dir, args.output, args.profile, args.seed, args.folds, args.tracking_uri)
    else:
        from .inference import load_bundle, predict
        print(json.dumps(predict(load_bundle(args.artifacts), read_csv(args.csv.read_bytes())), indent=2))


if __name__ == "__main__":
    main()
