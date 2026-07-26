"""
scripts/train.py

Train the pinned final CariSurg ESI triage model.

Run from the repository root:

    python scripts/train.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Add the repository root to Python's import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import (
    load_raw_data,
    clean_triage_data,
    get_features_and_target,
    split_data,
)

from src.features import add_clinical_features

from src.model import (
    build_logistic_regression,
    evaluate_model,
)


def main(config_path: str) -> None:
    # ---------------------------------------------------------------
    # 1. Load configuration
    # ---------------------------------------------------------------
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    random_seed = config["random_seed"]
    test_size = config["test_size"]

    raw_path = ROOT / config["data"]["raw_path"]

    print(f"Loading raw data from: {raw_path}")

    # ---------------------------------------------------------------
    # 2. Load and clean data
    # ---------------------------------------------------------------
    df_raw = load_raw_data(str(raw_path))
    df_clean = clean_triage_data(df_raw)

    # ---------------------------------------------------------------
    # 3. Select features and target
    # ---------------------------------------------------------------
    X, y = get_features_and_target(df_clean)

    # ---------------------------------------------------------------
    # 4. Reproduce the fixed train/test split
    # ---------------------------------------------------------------
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
    )

    # ---------------------------------------------------------------
    # 5. Apply Week 7 engineered clinical features
    # ---------------------------------------------------------------
    X_train = add_clinical_features(X_train)
    X_test = add_clinical_features(X_test)

    # ---------------------------------------------------------------
    # 6. Build the pinned final Logistic Regression model
    # ---------------------------------------------------------------
    model_config = config["final_model"]["hyperparameters"]

    model = build_logistic_regression(
        max_iter=model_config["max_iter"],
        random_state=model_config["random_state"],
    )

    # ---------------------------------------------------------------
    # 7. Train
    # ---------------------------------------------------------------
    print("Training final Logistic Regression model...")

    model.fit(X_train, y_train)

    # ---------------------------------------------------------------
    # 8. Evaluate
    # ---------------------------------------------------------------
    predictions = model.predict(X_test)

    metrics = evaluate_model(y_test, predictions)

    print("\nFinal model evaluation:")
    print("-----------------------")

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
