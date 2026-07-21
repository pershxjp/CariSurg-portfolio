# Decision Journal — Week 7 Model Choice

## Context

- The Week 6 Logistic Regression baseline (macro-F1 0.492) passed Dr. Reyes' review and became the bar every candidate model had to clear.
- The ED Board asked for a six-axis benchmark to determine whether added model complexity provides enough benefit in performance to justify its cost in training time, inference time and interpretability.

## Alternatives Considered

- Random Forest, both default settings and hyperparameter-tuned via 3-fold cross-validation (macro-F1 0.390 and 0.475 respectively).
- Gradient Boosting (`HistGradientBoostingClassifier`, macro-F1 0.416).
- Small MLP neural network (macro-F1 0.498, statistically tied with the baseline).

## Decision

I chose **Logistic Regression** as the recommended Phase 3 model because none of the alternatives improved macro-F1 by a margin large enough to offset their substantially higher training cost (up to 207.77 s vs 2.59 s) and lower interpretability.

## Reasoning

- **Performance:** Logistic Regression's macro-F1 (0.492) was at or near the top of all five models tested, effectively tied with the Small MLP (0.498) and ahead of both Random Forest variants and Gradient Boosting.
- **Operational cost:** the tuned Random Forest took roughly 80x longer to train (207.77 s vs 2.59 s) and ~22x longer per inference (0.0541 ms vs 0.0024 ms) for a lower macro-F1 than the baseline, so the extra cost bought nothing on the headline metric.
- **Interpretability and failure mode:** Logistic Regression's coefficients can be read directly (High interpretability), whereas the untuned Random Forest — despite the highest raw accuracy (0.641) — had an ESI 1 recall of 0.000, a hidden failure on the most critical patients that accuracy alone did not reveal.

## Things I Do Not Yet Know

- Whether Logistic Regression's own weak point, ESI 1 recall (0.250, the lowest of any model that didn't fail outright), can be fixed with a recalibrated threshold or safety-net rule without a full model change.
- Whether these results, measured on one train/test split on development hardware, will hold up under external validation, on hospital production hardware, or as the patient population drifts over time.
