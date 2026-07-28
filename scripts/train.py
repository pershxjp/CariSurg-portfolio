#!/usr/bin/env python3
"""
scripts/train.py
=================

Config-driven training entry point for the ESI triage project.

Usage:
    python scripts/train.py --config config.yaml

Everything this script does -- which raw file to read, which cleaning
pipeline to use, which model and hyperparameters to train, the random
seed, the test split -- comes from the config file. Nothing about the
pipeline's behaviour is hardcoded here; if you need to change any of
that, edit config.yaml, not this file.

What it does, in order:
    1. Load config.yaml.
    2. Load the raw CSV (config: data.raw_path).
    3. Clean it with the CANONICAL pipeline (src.data.clean_triage_data)
       -- NOT the legacy leaky replica. See src/data.py's module
       docstring and docs/model-selection.md for why that distinction
       matters.
    4. Optionally add engineered clinical features, if
       config.final_model.feature_set == "week7_engineered".
    5. Select features/target, split train/test (config: random_seed,
       test_size).
    6. Build the model named in config.final_model.type, with the
       hyperparameters listed there (src.model.build_model_from_config).
    7. Fit, evaluate (src.model.evaluate_model), print the results.
    8. Save the fitted model to disk (config: output.model_path, if set).

IMPORTANT: any metrics this script prints are a NEW run, not the
historical Week 6/7 notebook numbers. See docs/model-selection.md for
why those numbers can't be reproduced by this script (the leak finding).
Log new results as a new, dated entry -- don't silently overwrite the
historical table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/train.py` from the repo root without
# installing the package -- adds the repo root (parent of this file's
# directory) to sys.path so `import src...` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import (
    load_raw_data,
    clean_triage_data,
    select_feature_columns,
    get_features_and_target,
    split_data,
)
from src.features import add_clinical_features
from src.model import build_model_from_config, evaluate_model
from src.utils import load_config, get_logger, save_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ESI triage model from a config file.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML config file (default: config.yaml).",
    )
    return parser.parse_args()


def run(config_path: str) -> dict:
    """
    Run the full pipeline from a config file path. Returns the metrics
    dict from evaluate_model(). Split into its own function (rather than
    living directly under `if __name__ == '__main__'`) so tests can call
    it directly without shelling out to a subprocess.
    """
    logger = get_logger()
    config = load_config(config_path)

    logger.info("Loaded config from %s", config_path)

    random_seed = config["random_seed"]
    test_size = config["test_size"]
    final_model_config = config["final_model"]
    feature_set = final_model_config.get("feature_set", "plain")

    # 1. Load raw data.
    raw_path = config["data"]["raw_path"]
    logger.info("Loading raw data from %s", raw_path)
    df_raw = load_raw_data(raw_path)

    # 2. Clean with the CANONICAL pipeline -- never the legacy leaky one.
    logger.info("Cleaning data with the canonical Week 5 pipeline")
    df = clean_triage_data(df_raw)

    # 3. Optionally add engineered clinical features.
    if feature_set == "week7_engineered":
        logger.info("Adding Week 7 engineered clinical features")
        df = add_clinical_features(df)
    elif feature_set not in ("plain", "week7_engineered"):
        logger.warning(
            "Unrecognised feature_set '%s' in config -- proceeding with the plain feature set.",
            feature_set,
        )

    # 4. Select features/target and split.
    feature_columns = select_feature_columns(df)
    X, y = get_features_and_target(df, feature_columns)
    logger.info("Using %d features; %d rows before split", len(feature_columns), len(df))

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_seed)
    logger.info("Split: %d train / %d test", len(X_train), len(X_test))

    # 5. Build, fit, evaluate the model named in config.yaml.
    model = build_model_from_config(final_model_config)
    logger.info("Training %s", final_model_config["type"])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    logger.info(
        "Results -- accuracy: %.3f | macro_f1: %.3f | weighted_f1: %.3f | recall_esi1: %.3f",
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["weighted_f1"],
        metrics["recall_esi1"],
    )

    # 6. Save the fitted model, if a path is configured.
    model_path = config.get("output", {}).get("model_path")
    if model_path:
        saved_path = save_model(model, model_path)
        logger.info("Saved trained model to %s", saved_path)
    else:
        logger.info("No output.model_path in config -- model not saved to disk.")

    return metrics


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
