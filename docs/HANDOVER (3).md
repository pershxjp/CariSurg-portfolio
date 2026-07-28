# Handover Document

**Status:** Week 8 final submission. Written so a first-time reader can
clone this repository, follow the instructions below, and run the pinned
model the same day -- provided they have obtained the dataset through the
approved access process (see Section (d)).

---

## (a) Project summary

This project predicts a patient's Emergency Severity Index (ESI 1-5,
1 = most urgent) at triage, using vitals, chief-complaint flags, and
engineered clinical features, for the Mercer General Hospital emergency
department (Yale EMMLC dataset). A Phase 3 model direction has been
proposed (Logistic Regression) following six weeks of exploratory work
across Weeks 5-7, but as of this submission that recommendation rests on
a headline metric measured on uncleaned data with a leaked feature (see
Section (e), Limitation 1) and has not yet been re-validated on correctly
cleaned data -- it should be treated as provisional, not deployed, until
that re-run happens.

## (b) Final model decision

**Decision:** Logistic Regression (scaled, `max_iter=1000`,
`random_state=42`), per `config.yaml`'s `final_model` section.

**One-sentence reason:** among every model actually measured across
Weeks 6-7, Logistic Regression is tied for the best macro-F1 while being
far cheaper to train and run and fully interpretable via its
coefficients, whereas the alternatives either scored lower or hid a 0.000
ESI-1 recall behind higher raw accuracy.

Full reasoning is in the
[Week 7 decision journal](decision-journal/2026-week-7-model-choice.md)
and the complete results comparison is in
[`docs/model-selection.md`](model-selection.md). **Read the "Chosen
model" caveat there before treating this as final** -- see Limitation 1
below.

## (c) How to run the pipeline

Once you have the dataset (see Section (d)) placed at
`data/raw/yaleemmlc_admissionprediction_triage.csv`:

```bash
pip install -r requirements.txt
python scripts/train.py --config config.yaml
```

This loads the raw CSV, cleans it with the canonical pipeline
(`src.data.clean_triage_data`), optionally applies engineered clinical
features (`src.features.add_clinical_features`, controlled by
`config.yaml`'s `final_model.feature_set`), splits train/test
(`random_state=42`, stratified by `esi`, per `config.yaml`), trains the
model named in `config.yaml`, prints accuracy/macro-F1/weighted-F1/ESI-1
recall, and saves the fitted model to `models/final_model.joblib`.

To run the test suite:

```bash
pytest
```

## (d) Where the data lives, and its governance status

- **Location (current, live):** Google Drive, used inside Colab --
  `/content/drive/MyDrive/CariSurg/yaleemmlc_admissionprediction_triage.csv`
  for the raw export; `triage_cleaned_v1.csv` for the Week 6 pre-cleaned
  version. Confirmed by the project owner as the correct current path.
- **Location (repo convention, going forward):** `config.yaml` points at
  `data/raw/yaleemmlc_admissionprediction_triage.csv` (repo-relative).
  `data/processed/` is reserved for the cleaned version -- confirmed that
  `triage_cleaned_v1.csv` is exactly `src.data.clean_triage_data()`'s
  output on the raw file.
- **The CSVs themselves are git-ignored** (`.gitignore`) and must never
  be committed to this repository -- this is de-identified/sensitive
  clinical data. `data/raw/README.md` and `data/processed/README.md`
  flag this and point back here.
- **Sensitivity:** real (or realistically modelled) patient triage data
  -- vitals, demographics including race/ethnicity, and ESI outcome --
  from the Yale EMMLC dataset via Mercer General Hospital. Treat as
  PHI-adjacent unless confirmed otherwise.
- **Governance status: `NR` (Not Recorded)** in every artefact reviewed
  for this project. No notebook, report, or journal entry states a data
  use agreement, IRB/ethics approval reference, consent basis, or access
  control list.
  - [ ] TODO: confirm and document the data use agreement / IRB approval
        covering this dataset's use in model training.
  - [ ] TODO: confirm and document the actual data-access process (who
        grants access, what agreement is required) -- currently only
        "ask the project owner directly" (see `data/raw/README.md`).
  - [ ] TODO: confirm retention policy for the raw/cleaned CSVs and
        trained model artefacts (`.joblib` files).
  - **Do not treat this as resolved until the above are filled in by
    someone with authority over the data agreement** -- it is listed
    here so it isn't lost, not because it has been checked.

## (e) Known limitations

**1. The recommended model's headline metric was measured on uncleaned
data, including a leaked feature.** Week 6 Tutorial 2 -- the notebook
behind the 0.495 macro-F1 / 0.667 accuracy figure the Phase 3
recommendation primarily rests on -- loaded the raw CSV directly despite
its own comment claiming it used cleaned data, and trained on it
including the raw row-index column (`"Unnamed: 0"`) as a literal model
feature. It is not yet known whether Logistic Regression still leads the
field once trained on correctly cleaned data. See `src/data.py`'s module
docstring ("CRITICAL FINDING") and `docs/model-selection.md` for the full
write-up; closing this requires a new, verified run on
`src.data.clean_triage_data()` output, which requires the actual dataset
(not available in this development environment).

**2. Several historical metrics are simply not recorded anywhere, and
will not be reconstructed.** Precision, recall, accuracy, training time,
and inference time are marked `NR` (Not Recorded) throughout
`docs/model-selection.md` for most Week 7 models, because the source
notebook never computed or saved them -- not because they were lost.
Similarly, the tuned Random Forest's winning hyperparameters were never
saved. These gaps can only be closed by a fresh, dated re-run of the
actual code, never by estimating or backfilling a plausible-looking
number.

**3. No external validation has been performed.** Every metric in this
project comes from a single train/test split (80/20, stratified by
`esi`, `random_state=42`) on development hardware, on one snapshot of the
Yale EMMLC dataset. Nothing has been checked against a held-out
hospital, a different time period, or a production environment, and
ESI-1 recall (0.25 for Logistic Regression, itself measured on the
uncleaned data from Limitation 1) is the specific weak point most worth
scrutiny before any clinical use, given the safety cost of missing the
most urgent patients.

---

## Appendix: repository layout

- `src/data.py` -- loading, cleaning (canonical + clearly labelled
  legacy/deprecated variant), feature *selection*, splitting.
- `src/features.py` -- engineered clinical features
  (`add_clinical_features`).
- `src/model.py` -- model builders, the config-driven `MODEL_BUILDERS`
  registry, and `evaluate_model()`.
- `src/utils.py` -- config loading, model save/load, logging.
- `scripts/train.py` -- the entry point described in Section (c).
- `config.yaml` -- selected model, hyperparameters, seed, paths.
- `requirements.txt` -- pinned dependency versions.
- `tests/` -- schema test + training smoke test (`pytest` from repo
  root).
- `data/raw/`, `data/processed/` -- git-ignored; see Section (d).
- `docs/model-selection.md` -- full results table, the critical
  raw-data/leak finding, and open discrepancies.
- `docs/decision-journal/` -- the original Week 7 decision journal.
- `notebooks/` -- original Week 5-7 exploratory notebooks, kept as-is
  (all three Week 5 notebooks are included; Tutorial 3 contains the
  canonical `clean_triage()` function `src/data.py` is built from).

## Appendix: fuller backlog (superset of Section (e))

- Confirmed with the project owner: there is no Week 7 Tutorial 3
  notebook or other source to locate for the decision journal's
  six-axis benchmark (accuracy, ESI-1 recall, training/inference time,
  interpretability) -- those cells stay `NR` permanently unless someone
  re-runs the original code and captures fresh output.
- Decision journal vs. notebook macro-F1 mismatch for Random Forest
  (default), Gradient Boosting, and especially the MLP (0.498 vs. 0.449)
  -- resolved in favour of the notebook's saved output per project owner
  direction; the journal's numbers are documented but not used. See
  `docs/model-selection.md`.
- Logistic Regression has never been run on the engineered feature set
  that `config.yaml` currently selects as final, on top of canonical
  cleaning -- needs a new verified run, not a recovered value.

## Appendix: ethical / fairness notes to carry forward

- `race` and `ethnicity` are in `DEMOGRAPHICS` and excluded from the
  base feature set by default. Week 7 Tutorial 2, Cells 31-35, walks
  through one-hot encoding them back in and explicitly asks the student
  to "train a RandomForestClassifier on X_train_plus... and compare its
  macro-F1 to the demographics-free forest... did the extra features
  actually help?" -- **but Cell 35's own output was never saved**, so
  there is no recorded macro-F1 comparison to report here. This isn't a
  missing write-up; the underlying result doesn't exist in any supplied
  artefact. If this comparison has been run since (outside the
  notebook), get the actual before/after macro-F1 numbers from whoever
  ran it -- do not estimate or infer them. Cell 32 also flags this
  explicitly as ethically sensitive: "a higher accuracy score is not
  enough to justify" including race/ethnicity, and doing so "unless you
  can defend the choice and governance signs off" -- this caveat should
  carry forward regardless of what any re-run shows.

## Appendix: contacts / reviewers

- **Confirmed (source: `Week6_Baseline_Report.docx`, addressee line):** Dr
  De Fretias and Dr Marcus Reyes (Consultant Emergency Physician) were the
  named recipients of the Week 6 baseline report, and the decision
  journal confirms the Week 6 Logistic Regression baseline "passed Dr.
  Reyes' review" before becoming the bar every Week 7 candidate had to
  clear. Treat these two as the working assumption for who reviews the
  Phase 3 model choice, pending explicit confirmation of their formal
  sign-off role (report addressee and review-in-practice aren't
  necessarily the same as a named approver of record).
- **Still open:** the decision journal separately mentions "the ED
  Board" requesting the six-axis benchmark -- no named individuals for
  that board appear in any supplied source. If sign-off needs to go
  through the ED Board formally, get those names from the project
  owner; they cannot be inferred from what's been reviewed here.
