"""
src/data.py
===========

Data loading, cleaning, feature selection, and train/test splitting for
the Mercer General Hospital ESI triage project.

INTERIM NOTE (Week 8 refactor):
This module was extracted from `Week7_Tutorial2_Optimization_Techniques_STUDENT.ipynb`,
NOT from a Week 5 notebook, because no Week 5 cleaning notebook was supplied
during this refactor. Week 7 Tutorial 2 contains an inline comment stating
this cleaning block is "the SAME cleaning you did in Week 5, in compact
form" -- but that claim has not been independently verified against the
original Week 5 code. If your real Week 5 notebook differs from this
logic in ANY way (different imputation, different impossible-vital
cutoffs, different columns dropped), this file is WRONG and must be
corrected against that original source.
    TODO(owner): confirm this cleaning logic against the actual Week 5
    notebook and remove this note once verified.

Every function below is a direct, uncondensed translation of a specific
notebook cell. Nothing here changes the columns used, the imputation
strategy, the split ratio, or the random seed. Where the notebook made an
implicit choice (e.g. median imputation for missing vitals), that choice
is preserved exactly, not "improved."

Cell provenance (Week7_Tutorial2_Optimization_Techniques_STUDENT.ipynb):
    - clean_triage_data()       <- Cell 7  ("Load and clean the raw data")
    - select_feature_columns()  <- Cell 10 ("Choose the features (X) and the target (y)")
    - split_data()              <- Cell 11 ("Reproduce the EXACT Week 6 split")
    - add_clinical_features()   <- Cell 22 ("Feature engineering")

FINAL submission note: `add_clinical_features` will move to `src/features.py`
once that module is created. It lives here for now because the interim
scope for this submission is limited to `src/data.py` and `src/model.py`.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Column groups (verbatim from Week 6 Tutorial 2, Cell 7/8, and reused
# identically in Week 6 Tutorial 3 Cell 7 and Week 7 Tutorial 2 Cell 10).
# ---------------------------------------------------------------------------

TARGET = "esi"

# Vital-sign columns measured at the front door:
VITALS = [
    "triage_vital_hr",
    "triage_vital_sbp",
    "triage_vital_dbp",
    "triage_vital_rr",
    "triage_vital_o2",
    "triage_vital_temp",
    "triage_glucose",
]

# Who the patient is (fairness-sensitive -- handle with care; see the
# Week 7 Tutorial 2 "DATA / SAFETY NOTE" on race/ethnicity as inputs).
DEMOGRAPHICS = [
    "age",
    "gender",
    "ethnicity",
    "race",
    "lang",
    "religion",
    "maritalstatus",
    "employstatus",
    "insurance_status",
]

# Administrative / arrival details:
ADMIN = ["dep_name", "arrivalmode", "arrivalmonth", "arrivalday", "arrivalhour_bin"]

# Outcomes of the visit -- known only AFTER triage, so they must NEVER be
# model inputs (this is the data-leakage guard from every notebook).
LEAKAGE = ["disposition", "previousdispo"]


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw triage export exactly as read in Week 7 Tutorial 2, Cell 5.

    Parameters
    ----------
    path : str
        Path to the raw CSV (e.g. yaleemmlc_admissionprediction_triage.csv).
        TODO(owner): the notebooks read this from a hardcoded Google Drive
        path (/content/drive/MyDrive/CariSurg/...). config.yaml's
        `data.raw_path` should replace that hardcoded path -- confirm the
        real path once this repo is running outside Colab.
    """
    df_raw = pd.read_csv(path)
    return df_raw


def clean_triage_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the raw, messy export into a modelling-ready table.

    Direct translation of Week 7 Tutorial 2, Cell 7. Steps, in the exact
    order the notebook applies them (order matters -- e.g. impossible
    vitals are blanked BEFORE median imputation, so they contribute to
    neither):

    1. Drop stray index columns (e.g. "Unnamed: 0") that pandas adds on
       export/import -- not real data.
    2. Coerce vitals to numeric; unparseable text (e.g. "120bpm") -> NaN.
    3. Coerce ESI to numeric and drop any row without a valid ESI in
       {1, 2, 3, 4, 5} -- a row with no valid triage label cannot teach
       a triage model.
    4. Blank out physically impossible vitals (temp outside 90-110 F,
       SpO2 > 100) so they don't poison downstream imputation.
    5. Encode gender to 0/1, tolerant of odd casing ("m", "MALE", etc.).
    6. Fill remaining missing numeric values (vitals, age, gender) with
       the column median. This is a simple, defensible choice already
       made in the notebook -- NOT changed here.

    Do not add new cleaning steps or change cutoffs without confirming
    against the original Week 5 source (see module docstring TODO).
    """
    df = df_raw.copy()

    # 1) drop any stray index column -- it is not real data
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    # 2) force the vitals to be NUMBERS; unparseable text becomes NaN
    for col in VITALS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3) the ESI label must be 1-5. Drop rows where it is missing or out of range.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin([1, 2, 3, 4, 5])].copy()

    # 4) blank out physically impossible vitals so they don't poison the model
    df.loc[(df["triage_vital_temp"] < 90) | (df["triage_vital_temp"] > 110), "triage_vital_temp"] = np.nan
    df.loc[df["triage_vital_o2"] > 100, "triage_vital_o2"] = np.nan

    # 5) encode gender to 0/1 (handles odd casings like "m" / "MALE")
    df["gender"] = (
        df["gender"].astype(str).str.strip().str.lower().map({"male": 0, "m": 0, "female": 1, "f": 1})
    )

    # 6) fill remaining missing NUMBERS with the column median
    for col in VITALS + ["age", "gender"]:
        df[col] = df[col].fillna(df[col].median())

    df[TARGET] = df[TARGET].astype(int)
    return df


def add_clinical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered clinical features. Direct translation of Week 7
    Tutorial 2, Cell 22.

    IMPORTANT (see module docstring): as written in the notebook, this
    function was applied to X_train/X_test AFTER the train/test split
    and AFTER the baseline Logistic Regression was scored -- so no
    existing metric reflects Logistic Regression trained on this
    engineered feature set. Do not assume the Week 6/7 Logistic
    Regression macro-F1 figures (0.495 / 0.492) apply once this function
    is used; that combination has not been measured (see
    docs/model-selection.md TODO).

    Must be called identically on train and test data (fit-free -- these
    are deterministic transforms, not learned parameters, so there is no
    train/test leakage risk in applying it after the split).
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


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Choose which columns the model may use (X) as opposed to the target
    (y). Direct translation of Week 7 Tutorial 2, Cell 10 (identical to
    Week 6 Tutorial 2, Cell 7/8).

    Excludes the target, LEAKAGE columns (post-triage outcomes), ADMIN
    columns, and DEMOGRAPHICS columns from the base feature set -- exactly
    as in both source notebooks. (Week 7 Tutorial 2, Cells 31-35, separately
    explores ADDING one-hot-encoded demographics back in for comparison;
    that variant is NOT reproduced here -- see docs/model-selection.md.)
    """
    return [c for c in df.columns if c != TARGET and c not in LEAKAGE + ADMIN + DEMOGRAPHICS]


def get_features_and_target(df: pd.DataFrame, feature_columns: list[str] | None = None):
    """Split a cleaned dataframe into X (features) and y (target esi)."""
    if feature_columns is None:
        feature_columns = select_feature_columns(df)
    X = df[feature_columns]
    y = df[TARGET]
    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Reproduce the EXACT split used in every Week 6/7 notebook: stratified
    by ESI level, 20% held out, random_state=42.

    Do NOT change test_size or random_state here without updating
    config.yaml and re-running every downstream metric -- every reported
    number in docs/model-selection.md assumes this exact split.
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
