"""
src/data.py
===========

Data loading, cleaning, feature selection, and train/test splitting for
the Mercer General Hospital ESI triage project.

PROVENANCE (Week 8 refactor, updated after Week 5 notebooks were supplied):

    - clean_triage_data()          <- Week 5 Tutorial 3 (Exploratory
                                       Visualisation), Cell 8: `clean_triage(raw)`,
                                       explicitly commented "Canonical Week-5
                                       cleaning pipeline." This IS the ground
                                       truth cleaning function, verified
                                       against source.
    - clean_triage_data_legacy_week6_as_run()
                                    <- see "CRITICAL FINDING" below.
    - select_feature_columns()     <- Week 6 Tutorial 2, Cell 7-8; identical
                                       in Week 7 Tutorial 2, Cell 10.
    - split_data()                 <- Week 6 Tutorial 2, Cell 13; reproduced
                                       identically in Week 7 Tutorial 2, Cell 11.

See src/features.py for engineered clinical features (`add_clinical_features`,
moved there from an earlier draft of this file, Week 7 Tutorial 2, Cell 22).

CRITICAL FINDING (do not remove this note without team sign-off):
Week 6 Tutorial 2 -- the notebook that produced the 0.495 macro-F1 /
0.667 accuracy Logistic Regression figure your Phase 3 recommendation is
based on -- has a comment claiming it loads "the CLEANED dataset you
produced in Week 5 (triage_cleaned_v1.csv)", but the actual code on the
next line reads the RAW file by name:
    pd.read_csv('/content/drive/MyDrive/CariSurg/yaleemmlc_admissionprediction_triage.csv')
The saved output confirms this: 226 columns loaded, and the printed
feature list begins with 'Unnamed: 0' -- the raw pandas row-index
artifact was fed into Logistic Regression and the Decision Tree as a
real feature, with none of the cleaning below applied. No error was
raised, meaning this particular raw export happens to have no missing
values in the columns actually used as features -- but the index-column
leak is real, and neither Week 6 Tutorial 2 nor Week 6 Tutorial 3 (which
evaluates the same models) appears to have caught it.

Per project owner direction, this module now provides BOTH:
    1. `clean_triage_data()` -- the canonical, correct pipeline. This is
       the default and the one `config.yaml` / `scripts/train.py` should
       use going forward.
    2. `clean_triage_data_legacy_week6_as_run()` -- a clearly-labeled,
       deprecated function that faithfully reproduces what Week 6
       Tutorial 2 ACTUALLY ran (i.e. effectively no cleaning, including
       the 'Unnamed: 0' leak), kept ONLY so the historical 0.495/0.667
       figures remain traceable to actual code. Re-running the canonical
       pipeline will NOT reproduce those exact historical numbers -- see
       docs/model-selection.md for the full discussion.

Nothing in `clean_triage_data()` below changes the columns used, the
imputation strategy, the split ratio, or the random seed relative to the
verified Week 5 source. Where that source made an implicit choice (e.g.
median imputation only for vitals, not age), that choice is preserved
exactly, not "improved."

FINAL submission note: this module scope is now limited to loading,
cleaning, feature *selection* (which columns are model inputs), and
splitting. Engineered clinical features live in `src/features.py`.
"""

from __future__ import annotations

import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Column groups (verbatim from Week 5 Tutorial 3, Cell 5; identical in
# Week 6 Tutorial 2 Cell 7/8 and Week 7 Tutorial 2 Cell 10).
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

# Who the patient is (fairness-sensitive -- handle with care).
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

# The ~200 chief-complaint flag columns all share this prefix (Week 5
# Tutorial 3, Cell 5's classify_columns()).
CHIEF_COMPLAINT_PREFIX = "cc_"

# "Normal" adult triage reference ranges -- documentation/reference only,
# NOT used for cleaning decisions. Week 5 Tutorial 3, Cell 6. Temperature
# is Fahrenheit in this dataset.
NORMAL_RANGES = {
    "triage_vital_hr": (60, 100, "bpm"),
    "triage_vital_sbp": (90, 140, "mmHg"),
    "triage_vital_dbp": (60, 90, "mmHg"),
    "triage_vital_rr": (12, 20, "/min"),
    "triage_vital_o2": (95, 100, "%"),
    "triage_vital_temp": (97.0, 99.5, "F"),
    "triage_glucose": (70, 140, "mg/dL"),
}

# Values outside these bounds are treated as DATA ERRORS (e.g. a heart
# rate of 5) and blanked to NaN during cleaning -- much wider than
# "normal", these are physiological plausibility limits, not clinical
# targets. Week 5 Tutorial 3, Cell 6, verbatim.
PLAUSIBLE = {
    "age": (0, 120),
    "esi": (1, 5),
    "triage_vital_hr": (20, 250),
    "triage_vital_sbp": (50, 300),
    "triage_vital_dbp": (20, 200),
    "triage_vital_rr": (4, 60),
    "triage_vital_o2": (50, 100),
    "triage_vital_temp": (86, 110),
    "triage_glucose": (20, 800),
}


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw triage export.

    Parameters
    ----------
    path : str
        Path to the raw CSV (e.g. data/raw/yaleemmlc_admissionprediction_triage.csv,
        see config.yaml's `data.raw_path`). The notebooks read this from a
        hardcoded Google Drive path
        (/content/drive/MyDrive/CariSurg/yaleemmlc_admissionprediction_triage.csv);
        this parameter replaces that hardcoded path.

    Note on the stray index column: Week 5 Tutorial 3 loads the raw file
    with `pd.read_csv(path, index_col=0)`, which quietly discards the
    raw file's first (unnamed index) column into the DataFrame index.
    This function instead loads with plain `pd.read_csv(path)` and lets
    `clean_triage_data()` drop the resulting "Unnamed: 0" column
    explicitly -- functionally equivalent, but explicit rather than
    implicit, and importantly this is what makes the leak in
    `clean_triage_data_legacy_week6_as_run()` (below) visible: that
    legacy function deliberately skips the drop step to reproduce
    Week 6 Tutorial 2's actual behaviour.
    """
    df_raw = pd.read_csv(path)
    return df_raw


def _classify_columns(df: pd.DataFrame) -> dict:
    """
    Sort a DataFrame's columns into families. Direct translation of
    Week 5 Tutorial 3, Cell 5 (`classify_columns`).
    """
    def keep_present(wanted):
        return [c for c in wanted if c in df.columns]

    chief_complaints = [c for c in df.columns if c.startswith(CHIEF_COMPLAINT_PREFIX)]

    return {
        "target": keep_present([TARGET]),
        "vitals": keep_present(VITALS),
        "demographics": keep_present(DEMOGRAPHICS),
        "admin": keep_present(ADMIN),
        "leakage": keep_present(LEAKAGE),
        "chief_complaints": chief_complaints,
    }


def clean_triage_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the raw, messy export into a modelling-ready table.

    This is a direct, verified translation of the CANONICAL cleaning
    pipeline: Week 5 Tutorial 3 (Exploratory Visualisation), Cell 8,
    `clean_triage(raw)` -- confirmed via Cell 9 to be exactly what
    produces `triage_cleaned_v1.csv` (i.e. `config.yaml`'s
    `data.cleaned_path` = `clean_triage_data(load_raw_data(raw_path))`).

    Steps, in the exact order the source applies them:

    1. Drop the stray index column ("Unnamed: 0") -- equivalent to the
       source's `index_col=0` load (see load_raw_data docstring).
    2. Drop rows with no triage label at all (`esi` is NaN/missing) --
       a row with no valid triage label cannot teach a triage model.
       NOTE: unlike some other cleaning code you may have seen (e.g. an
       inline "compact recreation" in Week 7 Tutorial 2), this step does
       NOT separately filter esi to {1,2,3,4,5} by dropping rows -- that
       range check happens later, in step 4, via the shared PLAUSIBLE
       mechanism, exactly as the canonical source does it. In practice
       this means the canonical source implicitly assumes every non-null
       esi value in the raw file is already within 1-5 -- if that were
       ever not true, step 6 below (`.astype(int)`) would raise on the
       resulting NaN, exactly as it would in the original notebook. This
       is preserved faithfully, not "fixed" with a defensive filter that
       would silently behave differently.
    3. Make vitals (and age) numeric; unparseable text -> NaN.
    4. Blank out physiologically-impossible values to NaN, across EVERY
       column in `PLAUSIBLE` (age, esi, all 7 vitals) -- not just
       temperature and SpO2.
    5. Fill the gaps: vitals -> column median; a blank oxygen-device
       flag or chief-complaint flag means "not recorded" -> 0; blank
       demographic/admin/leakage TEXT columns -> the explicit category
       "Unknown". NOTE: `age` is deliberately NOT median-filled here --
       neither is it in the canonical source -- because `age` (like all
       of DEMOGRAPHICS) is excluded from the modelling features anyway
       (see select_feature_columns()), so an unfilled NaN there never
       reaches any model. This looks like a gap but is not one in
       practice; it is preserved as-is rather than "improved."
    6. The target must be a whole number 1-5, not a decimal.

    IMPORTANT: this function does NOT encode `gender` to 0/1. That
    encoding existed in an earlier draft of this file (copied from a
    Week 7 Tutorial 2 "compact recreation" of Week 5 cleaning that turned
    out not to match the real Week 5 source), but the canonical Week 5
    `clean_triage()` never does this -- `gender` is a DEMOGRAPHICS column
    and is simply text-filled like any other. Removing it changes
    nothing about any previously-reported metric, because DEMOGRAPHICS
    (including gender) is excluded from the base feature set in every
    Week 6/7 notebook regardless (see select_feature_columns()).
    """
    df = df_raw.copy()
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    fam = _classify_columns(df)

    # 1. Drop rows with no triage label.
    df = df[df[TARGET].notna()].copy()

    # 2. Make vitals (and age) numeric; unparseable text -> NaN.
    numeric_cols = list(fam["vitals"])
    if "age" in df.columns:
        numeric_cols.append("age")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Flag physiologically-impossible values as missing, across every
    #    PLAUSIBLE column (age, esi, all 7 vitals) -- not just temp/O2.
    for col, (low, high) in PLAUSIBLE.items():
        if col in df.columns:
            out_of_range = (df[col] < low) | (df[col] > high)
            df.loc[out_of_range, col] = np.nan

    # 4. Fill the gaps.
    for col in fam["vitals"]:
        df[col] = df[col].fillna(df[col].median())
    if "triage_vital_o2_device" in df.columns:
        df["triage_vital_o2_device"] = df["triage_vital_o2_device"].fillna(0)
    for col in fam["chief_complaints"]:
        df[col] = df[col].fillna(0)
    for col in fam["demographics"] + fam["admin"] + fam["leakage"]:
        if df[col].dtype == object:
            df[col] = df[col].fillna("Unknown")

    # 5. The target must be a whole number 1-5, not a decimal.
    df[TARGET] = df[TARGET].round().astype(int)
    return df


def clean_triage_data_legacy_week6_as_run(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    DEPRECATED -- do not use for any new work.

    Faithfully reproduces what Week 6 Tutorial 2 ACTUALLY ran when it
    produced the historical 0.495 macro-F1 / 0.667 accuracy Logistic
    Regression figures and the Decision Tree figures: essentially NO
    cleaning at all, despite that notebook's own comment claiming it
    loads cleaned data. Specifically:

    - No index column is dropped -- "Unnamed: 0" survives as a literal
      feature (see select_feature_columns(), which would happily include
      it, since it isn't in TARGET/LEAKAGE/ADMIN/DEMOGRAPHICS).
    - No missing-value handling, no implausible-value handling, no ESI
      validity filtering.
    - The only reason this ran without a scikit-learn "Input contains
      NaN" error is that this particular raw export apparently has no
      missing values in the columns that ended up as features -- that is
      an accident of this specific data export, not a property this
      function should be trusted to preserve for any other data.

    This function exists ONLY so the historical numbers in
    docs/model-selection.md remain traceable to real, inspectable code,
    for audit purposes. It must never be wired into config.yaml,
    scripts/train.py, or any new model-selection experiment. Calling it
    prints a loud warning for exactly this reason.
    """
    warnings.warn(
        "clean_triage_data_legacy_week6_as_run() reproduces a KNOWN DATA "
        "LEAK (raw row-index column included as a model feature) from the "
        "original Week 6 Tutorial 2 run. It exists only to keep the "
        "historical 0.495/0.667 Logistic Regression figures traceable to "
        "real code. Do not use it for any new training or evaluation.",
        stacklevel=2,
    )
    return df_raw.copy()


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Choose which columns the model may use (X) as opposed to the target
    (y). Direct translation of Week 6 Tutorial 2, Cell 7/8 (identical in
    Week 7 Tutorial 2, Cell 10).

    Excludes the target, LEAKAGE columns (post-triage outcomes), ADMIN
    columns, and DEMOGRAPHICS columns from the base feature set -- exactly
    as in both source notebooks. (Week 7 Tutorial 2, Cells 31-35, separately
    explores ADDING one-hot-encoded demographics back in for comparison;
    that variant is NOT reproduced here -- see docs/model-selection.md.)

    WARNING: if `df` still contains a raw index artifact column (e.g.
    "Unnamed: 0" -- see clean_triage_data_legacy_week6_as_run()), this
    function will include it as a feature, since it isn't in
    TARGET/LEAKAGE/ADMIN/DEMOGRAPHICS. That is the exact mechanism of the
    historical data leak documented in this module's docstring. Always
    run `clean_triage_data()` (not the legacy variant) before calling this
    for any new work.
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
