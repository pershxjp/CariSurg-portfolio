### carisurg-portfolio

A concise, clinician‑facing portfolio for the **CariSurg Healthcare AI Training Program**. This repository contains Week 0 exploratory data analysis notebooks and a Week 1 proposal exploring an **AI‑assisted emergency department triage** concept. No real patient data is stored in this repository.

---

### Overview

This repository documents early-stage analysis and a short proposal intended for clinical reviewers and healthcare professionals. Content is presented as Jupyter notebooks and a written proposal so reviewers can read findings and rationale without needing deep technical background.

---

### Repository Structure

- **`notebooks/`** — Week 0 exploratory data analysis notebooks (viewable in GitHub).
- **`proposal/`** — Week 1 proposal on AI‑assisted emergency department triage.
- **`README.md`** — This file.
- **`requirements.txt`** *(optional)* — Python package list for running notebooks locally.
- **`data/`** — Placeholder only; contains no real patient data.

---

### Project Objectives

- **Summarise Week 0 findings** from exploratory data analysis to highlight patterns relevant to triage.
- **Propose a Week 1 concept** for an AI‑assisted triage tool that supports clinical decision making in the emergency department.
- **Communicate clearly** to non‑technical clinical stakeholders the potential benefits, limitations, and next steps for safe development.

---

### Technologies Used

- **Jupyter Notebooks** for analysis and narrative.
- **Python** with common data libraries (pandas, matplotlib, seaborn).
- **Markdown** for readable proposals and documentation.

---

### How to Run the Notebooks

1. **View online** — Open notebooks directly on GitHub to read outputs and explanations.
2. **Run locally** (recommended for interactive use):
   - Install Python 3.8+.
   - Create a virtual environment and activate it.
   - Install dependencies:  
     `pip install -r requirements.txt`  
     *(If `requirements.txt` is not present, install pandas, matplotlib, seaborn, jupyter.)*
   - Start Jupyter:  
     `jupyter lab` or `jupyter notebook`
   - Open the notebooks in the `notebooks/` folder and run cells in order.
3. **No patient data** — the repository contains only synthetic or placeholder data; no special data governance is required to inspect files.

---

### Author Information

**Pershawn Jn Pierre** — Computer Information Technology student from Saint Lucia. Interests: **AI**, **healthcare technology**, and **robotics**.  

---
---
### Week 5 — Data Exploration & Feasibility (AI-Assisted Triage)

- **`notebooks/`** — three Week 5 notebooks:
  - Clinical Data Literacy — column-by-column walkthrough of the triage
    dataset (feature families, vitals reference ranges, chief-complaint
    structure, target definition).
  - Data Profiling — measures data types, missingness, distributions, and
    outliers to test whether the dataset is good enough to model.
  - Exploratory Visualisation — builds the data-quality dashboard
    (missingness, ESI class balance, demographics, chief complaints,
    vitals-by-ESI, correlation).
- **`docs/Feasibility Memo — Outline.pdf`** — draft feasibility memo answering
  whether the dataset is good enough to build a triage model on: verdict,
  dataset summary, top quality concerns, reasons to proceed, and caveats.
- **`docs/missingness_heatmap.png`** — missingness map of the structured
  columns before cleaning, referenced in the feasibility memo.
- **Dataset not committed.** The source CSV
  (`yaleemmlc_admissionprediction_triage.csv`) is a de-identified real
  clinical dataset, excluded via `.gitignore` due to file size and its
  clinical origin. It is not synthetic, and standard data-handling
  discipline (no raw rows pasted into external tools, summary statistics
  only) applies to this dataset even though it has been de-identified.

## Week 7 — Optimisation Techniques & Cost-Benefit Analysis (Model Complexity Decision)

* `notebooks/` — Week 7 Tutorial 2 notebook, *Optimization Techniques*: rebuilds the Week 6 Logistic Regression baseline, engineers clinical red-flag features (tachypnoea, hypoxia, fever, a combined red-flag count), trains and tunes a Random Forest (via `RandomizedSearchCV` with 3-fold cross-validation), compares performance with and without patient demographics, and benchmarks Gradient Boosting and a small MLP as further candidates.
* `docs/week-7-benchmark-table.md` — six-axis benchmark (accuracy, precision, recall, macro-F1, training time, inference time) across all five models, plus a seventh qualitative axis assessing whether a single prediction can be explained to a clinician in under a minute, and by what method (coefficients, feature importances/SHAP, or tree traversal).
* `docs/week-7-cost-benefit.md` and `docs/week-7-cost-benefit.pdf` — three-page cost-benefit memo for Dr. De Fretias, the ED Board, and Martina Griffith (Clinical IT Lead), recommending whether to carry a more complex model into Phase 3 or retain the Logistic Regression baseline, with arguments for and against, risks/unknowns, and a final recommendation.
* `docs/decisions/2026-week-7-model-choice.md` — decision journal entry recording the Phase 3 model choice, the concrete alternatives weighed, the benchmark-tied reasoning behind the decision, and what remains unknown.
