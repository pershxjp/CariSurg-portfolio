# Handover Document -- Outline (Draft, Week 8 Interim)

> One page target. This is a section-by-section outline with placeholder
> notes on what goes in each section -- not the final prose. Fill in bracketed
> items before the final submission.

## 1. Project summary (2-3 sentences)
- What the model predicts (ESI 1-5 at triage), for whom (Mercer General ED),
  and current status (Phase 3 recommendation made, not yet deployed).

## 2. Where things live
- `src/data.py` -- loading, cleaning, feature engineering, splitting.
- `src/model.py` -- model construction and evaluation.
- `config.yaml` -- selected model, hyperparameters, seed, paths (some TODO).
- `docs/model-selection.md` -- full results table + open discrepancies.
- `notebooks/` -- original Week 5-7 exploratory notebooks, kept as-is.
- [TODO, final submission] `src/features.py`, `src/utils.py`,
  `scripts/train.py`, `tests/`.

## 3. Where the data lives, and its governance status
- **Location (current, live):** Google Drive, used inside Colab --
  `/content/drive/MyDrive/CariSurg/yaleemmlc_admissionprediction_triage.csv`
  for the raw export; `triage_cleaned_v1.csv` for the Week 6 pre-cleaned
  version. Confirmed by the project owner as the correct current path.
- **Location (repo convention, going forward):** `config.yaml` now points
  at `data/raw/yaleemmlc_admissionprediction_triage.csv`
  (repo-relative), per the project owner's direction. `data/processed/`
  is reserved for the cleaned version once the Week 5 vs.
  `src.data.clean_triage_data()` question below is resolved.
- **The CSVs themselves are git-ignored** (`.gitignore`) and must never
  be committed to this repository -- this is de-identified/sensitive
  clinical data. `data/raw/README.md` and `data/processed/README.md`
  flag this and point back here; the actual access process (who to
  request the file from, what approval is needed) is still `TODO`.
- **Sensitivity:** this is real (or realistically-modeled) patient triage
  data -- vitals, demographics including race/ethnicity, and ESI outcome
  -- from the Yale EMMLC dataset via Mercer General Hospital. Treat as
  PHI-adjacent unless confirmed otherwise.
- **Governance status:** `NR` (Not Recorded) in every artifact reviewed
  for this refactor. No notebook, report, or journal entry states a data
  use agreement, IRB/ethics approval reference, consent basis, or access
  control list for this dataset.
  - [ ] TODO: confirm and document the data use agreement / IRB approval
        covering this dataset's use in model training.
  - [ ] TODO: confirm and document the actual data-access process
        (who grants access, what agreement is required) -- currently
        only "ask the project owner directly" (see `data/raw/README.md`).
  - [ ] TODO: confirm retention policy -- how long the raw/cleaned CSVs
        and trained model artifacts (`.joblib` files) should be kept.
  - **Do not treat this section as resolved until the above are filled
    in by someone with authority over the data agreement** -- it is
    listed here so it isn't lost, not because it has been checked.

## 4. How to reproduce the current results
- [TODO] exact command once `scripts/train.py` exists
  (interim: point to the relevant notebook cells listed in
  `src/data.py` / `src/model.py` docstrings).
- Fixed random seed: 42. Fixed split: 80/20 stratified by `esi`.

## 5. Current model recommendation
- Logistic Regression, per `docs/model-selection.md`, one-sentence reason:
  it's tied for the best macro-F1 actually measured while being far
  cheaper to train/run and fully interpretable, whereas the alternatives
  either scored lower or hid a 0.000 ESI-1 recall behind higher accuracy.
- **Flag prominently:** the specific configuration in `config.yaml`
  (Logistic Regression + Week 7 engineered features) has not yet been
  run/evaluated -- marked `NR` in that file and in model-selection.md,
  not a value that's simply missing from the paperwork.

## 6. Known gaps / unresolved items (do not let these get lost)
- [ ] Week 5 cleaning notebook was not available during this refactor;
      `src/data.py`'s cleaning logic is copied from Week 7 Tutorial 2's
      "compact recreation," which claims but does not prove equivalence
      to the real Week 5 pipeline.
- [ ] Confirmed with the project owner: there is no Week 7 Tutorial 3
      notebook or other source to locate. The six-axis benchmark
      (accuracy, ESI-1 recall, training time, inference time,
      interpretability) the decision journal describes was never
      supplied and isn't recoverable from an existing file -- those
      cells are marked `NR` in `docs/model-selection.md` permanently,
      unless someone re-runs the original code and captures fresh output.
- [ ] Decision journal vs. notebook macro-F1 mismatch for Random Forest
      (default), Gradient Boosting, and especially the MLP (0.498 vs
      0.449) -- resolved in favor of the notebook's saved output per
      project owner direction; the journal's numbers are documented but
      not used. See `docs/model-selection.md`.
- [ ] Tuned Random Forest's winning hyperparameters were never saved to
      any file; only the search space is reproducible from code.
- [ ] Logistic Regression has never been run on the engineered feature
      set that `config.yaml` currently selects as final -- needs a new
      verified run, not a recovered value.
- [ ] ESI-1 recall for Logistic Regression (0.25) is still the weakest
      point of the recommended model -- carried over from the decision
      journal's own "things I do not yet know" list.
- [ ] No external validation yet; single train/test split on
      development hardware.
- [ ] Data governance status (Section 3 above) is entirely unconfirmed --
      IRB/consent basis, access list, and retention policy all TODO.

## 7. Ethical / fairness notes to carry forward
- `race` and `ethnicity` are in `DEMOGRAPHICS` and excluded from the base
  feature set by default; Week 7 Tutorial 2 separately tested adding them
  back in (one-hot encoded) for comparison -- [TODO: summarize that
  finding here once confirmed, currently not reproduced in `src/data.py`].

## 8. Contacts / reviewers
- [TODO: Dr. Reyes, Dr. De Fretias, or others who should be listed as
  reviewers/approvers of the Phase 3 model choice.]
