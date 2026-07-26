# Model Selection -- Draft (Week 8 Interim)

> **Status:** DRAFT. Compiled only from metrics actually present, with saved
> output, in the uploaded notebooks and report. No number below has been
> invented, estimated, or backfilled from the decision journal where the
> journal and a notebook disagreed -- see "Known discrepancy" section.
>
> **Source confirmation (2026-08):** confirmed with the project owner that
> the Week 6 baseline notebook, the Week 7 Tutorial 2 notebook with saved
> outputs, and the Week 7 decision journal are the complete set of source
> artifacts for this refactor -- there is no Week 7 Tutorial 3 notebook or
> other dataset to locate. Metrics not present in that source set are
> marked `NR` (Not Recorded) below, permanently, not as a placeholder
> pending a future upload. See "Metric availability note" at the end of
> this document.

## Dataset and split (constant across all models)

- Source: Yale EMMLC triage dataset, n = 55,121 patients as loaded
  (Week 6 Tutorial 2, Cell 5). The post-cleaning row count is not
  separately recorded with saved output in the provided notebooks --
  marked `NR`, not estimated.
- Split: 80% train / 20% test, **stratified by `esi`**, `random_state=42`.
- Test set size: 11,025 patients (Week 6 Tutorials 2 & 3, confirmed).
- Target: `esi` (Emergency Severity Index, 1 = most urgent, 5 = least urgent).

## Week 6 results (plain feature set, 209 columns)

Source: `Week6_Tutorial2_Implement_LR_and_DT_STUDENT.ipynb`,
`Week6_Tutorial3_Model_Evaluation_STUDENT.ipynb`, and
`Week6_Baseline_Report.docx` (all three agree on these numbers).

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | ESI-1 Recall | Notes |
|---|---|---|---|---|---|---|---|
| Dummy (stratified random guess) | 0.375 | NR | NR | 0.204 | -- | 0.00 | Floor. `classification_report` was never run for this model. |
| **Logistic Regression** | 0.667 | 0.594 | 0.463 | **0.495** | 0.661 | 0.25 | Scaled; `max_iter=1000`. Only model with a full `classification_report` on record (Week 6 Tutorial 3, Cell 21). |
| Decision Tree (depth 5) | 0.556 | NR | NR | 0.216 | -- | 0.00 | Unscaled; highest accuracy of the three but 0 ESI-1 recall. `classification_report` was never run for this model. |

Per-class breakdown for Logistic Regression (Week 6 Tutorial 3, Cell 21 /
Baseline Report Table 2):

| ESI level | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 1 (most urgent) | 0.500 | 0.250 | 0.333 | 16 |
| 2 | 0.716 | 0.608 | 0.658 | 3,585 |
| 3 | 0.660 | 0.758 | 0.706 | 5,402 |
| 4 | 0.609 | 0.588 | 0.598 | 1,779 |
| 5 (least urgent) | 0.482 | 0.111 | 0.181 | 243 |

## Week 7 results (engineered feature set, see caveat below)

Source: `Week7_Tutorial2_Optimization_Techniques_STUDENT.ipynb`, Cell 54
(the only cell in that notebook with saved numeric output for these five
models).

| Model | Macro F1 | Accuracy | Precision | Recall | ESI-1 Recall | Training time | Inference time |
|---|---|---|---|---|---|---|---|
| Baseline (Logistic Regression, plain features) | 0.492 | NR | NR | NR | NR | NR | NR |
| Random Forest (default) | 0.392 | NR | NR | NR | NR | NR | NR |
| Random Forest (tuned, RandomizedSearchCV, 3-fold CV) | 0.475 | NR | NR | NR | NR | NR | NR |
| Gradient Boosting (HistGradientBoostingClassifier) | 0.421 | NR | NR | NR | NR | NR | NR |
| Small MLP (64,32) | 0.449 | NR | NR | NR | NR | NR | NR |

All `NR` cells above are genuinely absent from the source notebook: Cell
54 of Week 7 Tutorial 2 (the only cell with saved output for these five
models) computes and prints macro-F1 only. No other cell in that notebook
runs `classification_report`, `accuracy_score`, `recall_score`, or timing
code for these five models with output preserved. This is not being
searched for further -- see the source confirmation note at the top of
this document.

**Per your direction, these are treated as authoritative** (the notebook's
own saved output), in preference to the decision journal's figures for
the same runs -- see the discrepancy note below.

**Caveat on this table:** the "Baseline (Logistic Regression, plain
features)" row is *not* on the same feature set as the final selected
model in `config.yaml`. It's included here because it's the only Logistic
Regression number this notebook actually produced. Logistic Regression
has never been evaluated on the full engineered feature set
(`shock_index`, `pulse_pressure`, `spo2_rr_ratio`, `is_tachypneic`,
`is_hypoxic`, `is_febrile`, `red_flag_count`) that `config.yaml` currently
selects for Phase 3. This is marked `NR` in `config.yaml`'s
`known_metrics` -- it's a genuinely new experiment, not a historical
number that failed to get recorded, so closing it requires a **new
verified run** (see "Re-running to close the NR gaps" below), not
retrieval from an existing source.

Random Forest, Gradient Boosting, and the MLP *were* trained on the
engineered feature set (`X_train_fe`), so their numbers above are on a
different (larger) feature set than the Week 6 Logistic Regression/tree
numbers, and than the still-missing "LR on engineered features" run.
This is a genuine apples-to-oranges gap in the underlying notebooks, not
an artifact of this refactor -- flagging rather than silently normalizing.

## Known discrepancy: decision journal vs. notebook output

`2026-week-7-model-choice.md` reports different macro-F1 figures for three
of the five Week 7 models:

| Model | Decision journal | Notebook saved output (used above) |
|---|---|---|
| Random Forest (default) | 0.390 | 0.392 |
| Random Forest (tuned) | 0.475 | 0.475 (matches) |
| Gradient Boosting | 0.416 | 0.421 |
| Small MLP | **0.498** | **0.449** |

The journal also cites accuracy, ESI-1 recall, training time (up to
207.77 s), and inference time (0.0541 ms) figures for a full "six-axis
benchmark" that does not appear, with saved output, in any notebook
supplied for this refactor. Week 7 Tutorial 2 itself says this fuller
benchmark is "Tutorial 3" -- but per the project owner, no such notebook
or other source dataset exists to locate; the notebooks, saved outputs,
and decision journal already provided are the complete artifact set.

**Resolution:** per the project owner's direction, the notebook's saved
output (Cell 54) is treated as authoritative wherever it conflicts with
the decision journal, and the journal's numbers for Random Forest
(default), Gradient Boosting, and the MLP are **not** used above. The
0.498 vs. 0.449 MLP gap, and the smaller Random Forest/Gradient Boosting
gaps, are left unreconciled and undocumented beyond this note -- they
cannot be resolved without a fresh, verified re-run of the original code
(see "Re-running to close the NR gaps" below), not by picking whichever
number looks better.

## Recommendation (per decision journal, pending a new verified run)

**Logistic Regression** remains the Phase 3 recommendation: it is at or
near the top of macro-F1 among everything actually measured, trains and
infers far faster than any ensemble/neural alternative, and is fully
interpretable via its coefficients -- while the untuned Random Forest,
despite the highest raw accuracy in earlier testing, had an ESI-1 recall
of 0.000, a safety-relevant failure that accuracy alone conceals.

This recommendation is **not yet fully validated** for the specific
configuration selected in `config.yaml` (Logistic Regression + engineered
features) -- see the note above. Until that run exists, treat
`config.yaml`'s `final_model.known_metrics` as `NR`/unmeasured, not as
production-ready numbers.

## Re-running to close the NR gaps

Every `NR` cell above can, in principle, be closed -- not by searching for
a missing file, but by re-running the exact code already captured in
`src/data.py` and `src/model.py` against the original data, with the same
`random_state=42` split, and capturing the output this time. That would
produce a **new, verified run**, clearly distinguished from a historical
number, and should be logged as such (e.g. a dated results file or an
updated row here noting "re-run 2026-XX-XX", not silently merged into the
existing table). This has not been done as part of this refactor -- doing
so requires access to the actual dataset, which is not available in this
environment.

## Open items carried into Week 8 (from the decision journal, unchanged)

- Whether Logistic Regression's ESI-1 recall (0.25, Week 6) can be
  improved via a recalibrated threshold or safety-net rule.
- Whether these results, from one train/test split on development
  hardware, hold up under external validation or population drift.

## Metric availability note

Precision and recall were not consistently preserved for all historical
Week 6-7 runs. Only Logistic Regression (Week 6) has a full
`classification_report` on record; every other model's precision/recall
was simply never computed, or was computed but not saved to output, in
the notebooks provided. Missing values are marked `NR` throughout this
document rather than reconstructed, estimated, or inferred from a related
metric. This table is a record of what the original experiments actually
produced, not a complete benchmark -- gaps are gaps, not errors to paper
over.
