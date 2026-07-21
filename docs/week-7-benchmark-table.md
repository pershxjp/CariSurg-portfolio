# Week 7 Model Benchmark

| Model | Accuracy | Precision (macro) | Recall (macro) | Recall — ESI 1 | Macro-F1 | Training Time | Inference Time | Interpretability |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Logistic Regression (baseline) | 0.667 | 0.582 | 0.463 | 0.250 | 0.492 | 2.59 s | 0.0024 ms/row | High |
| Random Forest | 0.641 | 0.469 | 0.369 | 0.000 | 0.390 | 47.39 s | 0.1337 ms/row | Medium |
| Random Forest (tuned) | 0.608 | 0.452 | 0.519 | 0.312 | 0.475 | 207.77 s | 0.0541 ms/row | Medium |
| Gradient Boosting | 0.550 | 0.410 | 0.547 | 0.312 | 0.416 | 8.65 s | 0.0072 ms/row | Medium-Low |
| Small MLP | 0.638 | 0.526 | 0.482 | 0.312 | 0.498 | 141.68 s | 0.0038 ms/row | Low |

*Recall — ESI 1 is reported alongside the six required axes because ESI 1 (most critical) patients are the ones a triage model can least afford to miss; a model can look strong on macro-F1 while quietly failing this group.*

## Summary

**Overall macro-F1:** the Small MLP (0.498) and the Logistic Regression baseline (0.492) are effectively tied at the top; the tuned Random Forest trails by about 0.017–0.023, and the untuned Random Forest and Gradient Boosting trail further still (0.39–0.42). No complex model produces a clear win over the baseline on the headline metric — none of the additional complexity paid for itself in macro-F1.

**Cost side of the ledger:** every complex model costs far more to train than the baseline. Random Forest (tuned) trains ~80× slower than Logistic Regression (207.77 s vs 2.59 s) and infers ~22× slower per row (0.0541 ms vs 0.0024 ms). The Small MLP trains ~55× slower for a macro-F1 gain of only 0.006 over the baseline — not a defensible trade.

**The one result that changes the conversation:** the untuned Random Forest has a Recall — ESI 1 of **0.000** — it misses every single most-critical patient in the test set, despite having the *highest raw accuracy* (0.641) in the table. This is the clearest illustration in this benchmark of why accuracy alone is a misleading metric for a triage model: a model can look "best" and still be clinically dangerous. Tuning fixed this (Recall — ESI 1 rises to 0.312), but only by trading away accuracy and precision, and at a large training-time cost.

**Bottom line:** the improvement from any complex model is not large enough, on this test set, to justify the extra training cost and reduced interpretability. Logistic Regression is not just cheaper and more explainable — it is competitive or better on the primary metric. The one legitimate argument for a tuned Random Forest is its improved ESI 1 recall (0.312 vs 0.250), which matters clinically even though it doesn't show up as a macro-F1 win; this is exactly the kind of trade-off Argument 2 (clinical usefulness) in the cost-benefit memo needs to address explicitly rather than being settled by the aggregate score.

## Interpretability Assessment

| Model | Method used | Explain one prediction to Dr. Reyes in <1 minute? |
|---|---|---|
| Logistic Regression | Read off the scaled coefficients for the predicted class; the sign and magnitude of each feature's contribution is exact, not approximate. | **Yes.** "This patient's predicted risk increased because [feature] pushed the score up by X, and [feature] pushed it down by Y." |
| Random Forest / Random Forest (tuned) | `feature_importances_` gives a global ranking; a single prediction requires either a SHAP TreeExplainer or manually tracing the majority vote across a sample of the 300 trees. | **Yes, but only with SHAP already set up.** Without it, describing *why this specific patient* triggered a high-risk vote across hundreds of trees cannot honestly be done in a minute — you can only speak in aggregate ("many trees independently agreed based on combinations of features"). |
| Gradient Boosting | `HistGradientBoostingClassifier` has no built-in `feature_importances_`; permutation importance or SHAP is needed for even a global ranking. | **No, not without tooling.** Sequential trees correcting each other's errors are harder to trace than a forest's vote, and there's no quick built-in importance to point to. |
| Small MLP | Weights connect every input to every hidden neuron with no direct clinical meaning; a real explanation needs a post-hoc method (SHAP/LIME) and even then is an approximation of the network's behaviour, not its mechanism. | **No.** The most honest one-minute answer for an MLP is "the network's internal math doesn't map onto any clinical concept," which is not an explanation Dr. Reyes can act on. |

**Governing question:** for Logistic Regression, "can I explain one prediction in under a minute?" is answered by the model itself. For every other model in this table, the honest answer is "only if I've already built a SHAP explainer" — which is itself an admission of added engineering and maintenance cost that belongs in the cost-benefit memo's "Against" arguments.
