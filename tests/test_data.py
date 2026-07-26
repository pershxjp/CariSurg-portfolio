"""
Tests for the data-loading and cleaning pipeline.
"""

import pandas as pd

from src.data import (
    TARGET,
    VITALS,
    clean_triage_data,
    get_features_and_target,
)


def make_small_raw_dataset() -> pd.DataFrame:
    """Create a small synthetic dataset for testing."""

    return pd.DataFrame(
        {
            "esi": [1, 2, 3, 4, 5],
            "triage_vital_hr": [80, 90, 100, 110, 120],
            "triage_vital_sbp": [120, 130, 140, 150, 160],
            "triage_vital_dbp": [80, 85, 90, 95, 100],
            "triage_vital_rr": [16, 18, 20, 22, 24],
            "triage_vital_o2": [98, 97, 96, 95, 94],
            "triage_vital_temp": [98.0, 98.6, 99.0, 100.0, 101.0],
            "triage_glucose": [90, 100, 110, 120, 130],
            "age": [20, 30, 40, 50, 60],
            "gender": ["male", "female", "m", "f", "male"],
        }
    )


def test_cleaned_data_has_expected_schema():
    """
    Sanity check: cleaning should preserve the target and vital-sign columns.
    """

    raw_data = make_small_raw_dataset()

    cleaned = clean_triage_data(raw_data)

    # Target column must exist
    assert TARGET in cleaned.columns

    # All expected vital-sign columns must exist
    for column in VITALS:
        assert column in cleaned.columns

    # ESI values must be valid
    assert cleaned[TARGET].isin([1, 2, 3, 4, 5]).all()


def test_get_features_and_target_returns_matching_shapes():
    """
    X and y must contain the same number of rows.
    """

    raw_data = make_small_raw_dataset()

    cleaned = clean_triage_data(raw_data)

    X, y = get_features_and_target(cleaned)

    assert len(X) == len(y)
    assert len(X) > 0
    assert TARGET not in X.columns
