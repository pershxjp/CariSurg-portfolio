"""
src/features.py
================

Engineered clinical features for the ESI triage project.

Direct translation of Week 7 Tutorial 2, Cell 22. Moved here (out of
src/data.py) for the final submission, per the project's target
structure -- no logic changed in the move, only the file it lives in.
"""

from __future__ import annotations

import pandas as pd


def add_clinical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered clinical features. Direct translation of Week 7
    Tutorial 2, Cell 22.

    IMPORTANT: as written in the notebook, this function was applied to
    X_train/X_test AFTER the train/test split and AFTER the baseline
    Logistic Regression was scored -- so no historical metric reflects
    Logistic Regression trained on this engineered feature set. Do not
    assume the historical Logistic Regression macro-F1 figures apply
    once this function is used; that combination is marked NR
    (Not Recorded) in config.yaml and docs/model-selection.md and
    requires a new verified run.

    Must be called identically on train and test data (fit-free -- these
    are deterministic transforms, not learned parameters, so there is no
    train/test leakage risk in applying it after the split).

    Expects `data` to already have been through
    `src.data.clean_triage_data()` (so the vitals columns are numeric and
    implausible values have already been handled) -- calling this on
    unclean/raw data will silently propagate whatever mess is already in
    the vitals columns into the engineered features below.
    """
    out = data.copy()

    # --- ratios & combinations ---
    out["shock_index"] = out["triage_vital_hr"] / out["triage_vital_sbp"]       # HR / SBP (uses BP)
    out["pulse_pressure"] = out["triage_vital_sbp"] - out["triage_vital_dbp"]   # SBP - DBP (uses BP)
    out["spo2_rr_ratio"] = out["triage_vital_o2"] / out["triage_vital_rr"]      # oxygen vs effort (no BP)

    # --- red-flag flags (no blood pressure needed) ---
    out["is_tachypneic"] = (out["triage_vital_rr"] > 20).astype(int)      # fast breathing
    out["is_hypoxic"] = (out["triage_vital_o2"] < 92).astype(int)         # low oxygen
    out["is_febrile"] = (out["triage_vital_temp"] >= 100.4).astype(int)   # fever

    # --- severity score = how many red flags fire ---
    out["red_flag_count"] = out[["is_tachypneic", "is_hypoxic", "is_febrile"]].sum(axis=1)

    return out


ENGINEERED_FEATURE_NAMES = [
    "shock_index",
    "pulse_pressure",
    "spo2_rr_ratio",
    "is_tachypneic",
    "is_hypoxic",
    "is_febrile",
    "red_flag_count",
]
