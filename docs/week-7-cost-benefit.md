# Cost-Benefit Memo: Phase 3 Model Selection for ED Triage

**To:** Dr. De Fretias; The ED Board; Martina Griffith, Clinical IT Lead
**From:** [Student], Clinical-AI Trainee Team
**Date:** Week 7
**Re:** Whether to carry a more complex model into Phase 3, or retain the Logistic Regression baseline

**Recommendation:** retain Logistic Regression as the Phase 3 primary model, because none of the three complex models tested — Random Forest, tuned Random Forest, or Gradient Boosting — nor the Small MLP produced an improvement in macro-F1 large enough, or consistent enough, to justify their materially higher training cost and reduced interpretability; a separate, targeted piece of work is still required to improve detection of the most critical patients (Triage Level ESI 1), which this recommendation does not by itself solve.

---

## 1. Dataset and Methods Recap

This assessment uses the same Emergency Department triage dataset introduced in Week 5 and re-used for the Week 6 baseline. Each row is one ED attendance; the front-door vital signs (heart rate, systolic and diastolic blood pressure, respiratory rate, oxygen saturation, temperature and glucose), chief-complaint flags and, where tested, patient demographics were used as inputs. The target variable is the Emergency Severity Index (ESI), a five-level clinical triage score from 1 (most critical, resuscitation) to 5 (least urgent). Outcome fields such as final disposition were deliberately excluded from the inputs, as they are only known after triage has already happened and would leak information the model would not have in real use.

After cleaning, the dataset was split 80/20 into training and test sets using the same random seed as Week 6, so that all models in this memo are compared on identical, previously unseen patients. Five candidate models were benchmarked: the Week 6 Logistic Regression baseline, a Random Forest, a hyperparameter-tuned Random Forest (selected by three-fold cross-validation over tree count, depth and leaf size), a Gradient Boosting classifier, and a small neural network (Multi-Layer Perceptron). All models were evaluated on the same test set using accuracy, macro-averaged precision, recall and F1-score (which treats every ESI level as equally important, including the rare but critical ESI 1), plus wall-clock training time and per-patient inference time. This memo keeps the technical detail here to a minimum; the notebook and code underlying these figures are available on request.

## 2. Benchmark Results

| Model | Accuracy | Precision (macro) | Recall (macro) | Recall — ESI 1 | Macro-F1 | Training Time | Inference Time | Interpretability |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Logistic Regression (baseline)** | 0.667 | 0.582 | 0.463 | 0.250 | 0.492 | 2.59 s | 0.0024 ms/patient | High |
| Random Forest | 0.641 | 0.469 | 0.369 | 0.000 | 0.390 | 47.39 s | 0.1337 ms/patient | Medium |
| Random Forest (tuned) | 0.608 | 0.452 | 0.519 | 0.312 | 0.475 | 207.77 s | 0.0541 ms/patient | Medium |
| Gradient Boosting | 0.550 | 0.410 | 0.547 | 0.312 | 0.416 | 8.65 s | 0.0072 ms/patient | Medium-Low |
| Small MLP | 0.638 | 0.526 | 0.482 | 0.312 | 0.498 | 141.68 s | 0.0038 ms/patient | Low |

What this means in practice: the Logistic Regression baseline is at or near the top of the table on the primary metric (macro-F1), while being by far the cheapest to train and the fastest and simplest to run and explain. The tuned Random Forest is the only complex model that reliably improves on detecting the most critical (ESI 1) patients, but it does so while scoring lower overall than the baseline, and at roughly 80 times the training time. The untuned Random Forest is the clearest cautionary result in this table: it has the joint-highest accuracy, yet its recall for ESI 1 patients is zero — it misses every single one of the most urgent patients in the test set. This is the strongest evidence in the whole exercise that a single headline accuracy figure can actively hide a dangerous model.

## 3. Three Arguments FOR Retaining Logistic Regression

**Argument 1 — Performance.** On macro-F1, the metric chosen specifically because it does not let common, low-acuity cases dominate the score, Logistic Regression (0.492) is statistically indistinguishable from the best-performing complex model, the Small MLP (0.498), and outperforms both Random Forest variants and Gradient Boosting. No complex model demonstrated a meaningful, reliable lift over the baseline on the metric the Board has asked us to optimise.

**Argument 2 — Predictable behaviour, no catastrophic failure mode.** Logistic Regression's recall for ESI 1 patients (0.250) is modest, but it is non-zero and consistent with its behaviour on every other category. By contrast, the untuned Random Forest — the model that would look most attractive on accuracy alone — fails completely on this category (recall 0.000). A simpler model that is honestly mediocre everywhere is a safer starting point than a more complex model that can silently fail on the group of patients where failure matters most.

**Argument 3 — Operational feasibility.** Logistic Regression trains in under three seconds and scores a patient in roughly two-thousandths of a millisecond, against training times of up to 207 seconds and inference times up to fifty-six times slower for the alternatives tested. It requires no specialist hardware, no GPU, and no additional explainability tooling: its coefficients are directly readable, so a member of the clinical IT team can retrain, redeploy and sanity-check it inside a single working session, without new infrastructure or new failure modes to govern.

## 4. Three Arguments AGAINST Retaining Logistic Regression

**Against 1 — Modest absolute performance.** A macro-F1 of 0.492 and macro-recall of 0.463 mean the model is, in absolute terms, still wrong for a substantial share of patients across the five ESI levels. Retaining the baseline is a decision to accept this level of error for now, not a claim that the problem is solved.

**Against 2 — Weakest ESI 1 recall among the models that don't fail outright.** Excluding the untuned Random Forest's complete failure, Logistic Regression has the lowest recall (0.250) for the most critical patients; the tuned Random Forest, Gradient Boosting and Small MLP all reach 0.312. For a triage tool, under-detecting the sickest patients is the single most consequential error type, and the baseline is not the strongest performer here.

**Against 3 — Linear models cannot capture compensatory or non-linear physiology.** Conditions such as early compensated shock can present with vitals that look individually unremarkable but combine into a dangerous pattern (this is exactly why the team engineered features such as the shock index). A purely linear model may under-weight these interactions relative to a tree-based model that can learn them directly, and this may partly explain its comparatively low ESI 1 recall.

## 5. Risks and Unknowns

- The training and test data may not represent the patient population the hospital will see in future — seasonal illness patterns, new patient demographics, or changes in coding practice could all shift the picture.
- None of the five models, including the recommended one, has been externally validated on a different hospital site or a genuinely held-out time period.
- The effect of data drift — vitals equipment changes, new triage protocols, coding changes — on any of these models over time is unknown and has not been tested.
- Training and inference times were measured on a development laptop, not the hospital's production hardware; relative rankings are likely to hold, but absolute figures will not transfer directly.
- A strong statistical score, on any axis, is not proof of clinical safety. Ongoing clinical audit and a human-in-the-loop review process for all model-assisted triage decisions remain necessary regardless of which model is deployed.
- The test set is a single split of a limited number of patients; the differences in macro-F1 between the top three models (Logistic Regression, Small MLP, tuned Random Forest) are small enough that they may not be statistically distinguishable without repeated cross-validation or confidence intervals, which have not yet been computed.
- The effect of including or excluding patient demographics (age, gender, ethnicity, race) on both performance and fairness has been explored only preliminarily in this phase and requires its own dedicated ethics and bias review before any production decision.

## 6. Final Recommendation

**Recommendation: retain Logistic Regression as the Phase 3 primary model.** None of the complex models tested — Random Forest, tuned Random Forest, Gradient Boosting or Small MLP — improved macro-F1 by a margin large enough to offset their additional training cost, additional inference cost, and reduced interpretability; two of the four (untuned Random Forest, Gradient Boosting) scored materially worse on the primary metric, and the strongest alternative (Small MLP) was only marginally ahead while being the least explainable model in the table and roughly fifty-five times slower to train.

This recommendation does **not**, on its own, solve the weakest point in the current system: Logistic Regression's recall for ESI 1, the most critical patients, is the lowest of any model that did not fail outright, at 0.250. Adopting the baseline defers rather than resolves this problem. We recommend a separate, narrowly scoped follow-up before Phase 3 sign-off, focused specifically on ESI 1 sensitivity — for example, a recalibrated decision threshold for the ESI 1 class, a rule-based safety-net check layered on top of the model's output, or further feature engineering targeted at early compensated-shock presentations — rather than a wholesale change of model family. The tuned Random Forest should be retained as a documented benchmark and fallback candidate, to be revisited if a future iteration demonstrates a clear and reproducible improvement in both macro-F1 and ESI 1 recall together.
