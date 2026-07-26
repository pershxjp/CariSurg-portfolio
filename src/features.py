
"""
src/features.py

Feature engineering for the Mercer General Hospital ESI triage project.
"""

from __future__ import annotations

import pandas as pd


def add_clinical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add deterministic clinical features to the triage dataset.

    These features were introduced in Week 7:
    - shock_index
    - pulse_pressure
    - spo2_rr_ratio
    - is_tachypneic
    - is_hypoxic
    - is_febrile
    - red_flag_count
    """

    out = data.copy()

    # Ratios and combinations
    out["shock_index"] = (
        out["triage_vital_hr"] / out["triage_vital_sbp"]
    )

    out["pulse_pressure"] = (
        out["triage_vital_sbp"] - out["triage_vital_dbp"]
    )

    out["spo2_rr_ratio"] = (
        out["triage_vital_o2"] / out["triage_vital_rr"]
    )

    # Clinical red-flag indicators
    out["is_tachypneic"] = (
        out["triage_vital_rr"] > 20
    ).astype(int)

    out["is_hypoxic"] = (
        out["triage_vital_o2"] < 92
    ).astype(int)

    out["is_febrile"] = (
        out["triage_vital_temp"] >= 100.4
    ).astype(int)

    # Count how many red flags are present
    out["red_flag_count"] = out[
        [
            "is_tachypneic",
            "is_hypoxic",
            "is_febrile",
        ]
    ].sum(axis=1)

    return out
