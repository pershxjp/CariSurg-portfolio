"""
Smoke test for the final Logistic Regression training pipeline.
"""

import pandas as pd

from src.data import (
    clean_triage_data,
    get_features_and_target,
    split_data,
)

from src.features import add_clinical_features

from src.model import (
    build_logistic_regression,
    evaluate_model,
)


def make_small_training_dataset() -> pd.DataFrame:
    """
    Create approximately 50 synthetic rows for a fast training smoke test.
    """

    rows = []

    for i in range(50):
        rows.append(
            {
                "esi": (i % 5) + 1,
                "triage_vital_hr": 70 + (i % 50),
                "triage_vital_sbp": 110 + (i % 40),
                "triage_vital_dbp": 70 + (i % 30),
                "triage_vital_rr": 12 + (i % 20),
                "triage_vital_o2": 90 + (i % 11),
                "triage_vital_temp": 97.0 + (i % 5),
                "triage_glucose": 90 + (i % 60),
                "age": 18 + (i % 70),
                "gender": "male" if i % 2 == 0 else "female",
            }
        )

    return pd.DataFrame(rows)


def test_final_model_training_smoke_test():
    """
    The final Logistic Regression pipeline should train and make predictions
    on a small dataset.
    """

    raw_data = make_small_training_dataset()

    # Clean data
    cleaned = clean_triage_data(raw_data)

    # Select features and target
    X, y = get_features_and_target(cleaned)

    # Reproduce the project split
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Apply the Week 7 engineered features
    X_train = add_clinical_features(X_train)
    X_test = add_clinical_features(X_test)

    # Build the pinned final model
    model = build_logistic_regression(
        max_iter=1000,
        random_state=42,
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Basic smoke-test assertions
    assert len(predictions) == len(y_test)
    assert len(predictions) > 0

    # Confirm evaluation works
    metrics = evaluate_model(y_test, predictions)

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "recall_esi1" in metrics
