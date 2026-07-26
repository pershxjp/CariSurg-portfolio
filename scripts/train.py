"""
Train the final CariSurg ESI triage model.

Run from the repository root:
    python scripts/train.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Make the repository root importable when running:
# python scripts/train.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import (
    load_raw_data,
    clean_triage_data,
    get_features_and_target,
    split_data,
)
from src.model import build_random_forest, evaluate_model


def main(config_path: str) -> None:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    seed = config["seed"]

    raw_path = ROOT / config["data"]["raw_path"]

    print(f"Loading data from: {raw_path}")

    df_raw = load_raw_data(raw_path)
    df_clean = clean_triage_data(df_raw)

    X, y = get_features_and_target(df_clean)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=seed,
    )

    model_config = config["model"]

    model = build_random_forest(
        n_estimators=model_config["n_estimators"],
        class_weight=model_config["class_weight"],
        random_state=seed,
        n_jobs=model_config["n_jobs"],
    )

    print("Training final model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = evaluate_model(y_test, predictions)

    print("\nModel evaluation:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args()
    main(args.config)
