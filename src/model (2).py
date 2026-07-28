"""
src/model.py
============

Model construction and evaluation logic for the ESI triage project.

Every builder function below reproduces the exact hyperparameters used in
the source notebook cell it's derived from -- nothing tuned, added, or
"cleaned up" beyond what was already there. Where a notebook's own output
was not saved (e.g. the tuned Random Forest's winning hyperparameters),
this file uses the documented SEARCH SPACE only, not a guessed winner --
see the note on `random_forest_search_space`.

Cell provenance:
    - build_dummy_baseline()            <- Week 6 Tutorial 2, Cell 17
    - build_logistic_regression()       <- Week 6 Tutorial 2, Cell 21
    - build_decision_tree()             <- Week 6 Tutorial 2, Cell 25
    - build_random_forest()             <- Week 7 Tutorial 2, Cell 28
    - build_random_forest_tuned_search()<- Week 7 Tutorial 2, Cell 39
    - build_gradient_boosting()         <- Week 7 Tutorial 2, Cell 43
    - build_mlp()                       <- Week 7 Tutorial 2, Cell 47
    - evaluate_model()                  <- Week 6 Tutorial 3, Cells 21, 26, 31, 33
"""

from __future__ import annotations

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, recall_score

RANDOM_STATE = 42  # fixed throughout every notebook -- do not change here alone


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_dummy_baseline(
    strategy: str = "stratified", random_state: int = RANDOM_STATE
) -> DummyClassifier:
    """
    Random-guess floor. Week 6 Tutorial 2, Cell 17.

    Defaults reproduce the notebook exactly; pass different values only
    for deliberate experimentation, not routine use.
    """
    return DummyClassifier(strategy=strategy, random_state=random_state)


def build_logistic_regression(
    max_iter: int = 1000, random_state: int = RANDOM_STATE
) -> Pipeline:
    """
    Scaled Logistic Regression. Week 6 Tutorial 2, Cell 21 (and rebuilt
    identically in Week 7 Tutorial 2, Cell 16, on the plain feature set).

    Defaults reproduce the notebook exactly.

    NOTE: this exact configuration has only ever been scored on the plain
    (non-engineered) feature set -- see src/data.py docstring and the
    "NR" entries in docs/model-selection.md / config.yaml if this is
    retrained on engineered features; that combination has not been
    measured yet.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, random_state=random_state),
    )


def build_decision_tree(
    max_depth: int = 5, random_state: int = RANDOM_STATE
) -> DecisionTreeClassifier:
    """
    Unscaled Decision Tree. Week 6 Tutorial 2, Cell 25.

    Defaults reproduce the notebook exactly (depth 5).
    """
    return DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)


def build_random_forest(
    n_estimators: int = 300,
    class_weight: str | None = "balanced",
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """
    Random Forest. Week 7 Tutorial 2, Cell 28.

    Defaults reproduce the notebook's "default-settings" run exactly.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def random_forest_search_space(
    n_estimators_options: list[int] | None = None,
    max_depth_options: list[int | None] | None = None,
    min_samples_leaf_options: list[int] | None = None,
    max_features_options: list[str | None] | None = None,
) -> dict:
    """
    The hyperparameter search space used for Random Forest tuning.
    Week 7 Tutorial 2, Cell 39. Defaults reproduce the notebook's search
    space exactly.

    NOTE: the notebook's own `search.best_params_` output was never saved
    (empty output cell in the source notebook), so the WINNING combination
    from the original run is not available in any provided artifact --
    confirmed with the project owner that this is not simply missing from
    what was uploaded. This function documents only the search space that
    was explored, not a recovered winner. Re-running
    `build_random_forest_tuned_search()` below with `random_state=42` and
    the same data/split should reproduce the same winner deterministically,
    but that is a NEW verified run, not a value retrieved from this refactor.
    """
    return {
        "n_estimators": n_estimators_options or [100, 200, 300, 400],
        "max_depth": max_depth_options or [None, 6, 10, 16],
        "min_samples_leaf": min_samples_leaf_options or [1, 2, 4, 8],
        "max_features": max_features_options or ["sqrt", "log2", None],
    }


def build_random_forest_tuned_search(
    param_distributions: dict | None = None,
    n_iter: int = 8,
    cv: int = 3,
    scoring: str = "f1_macro",
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> RandomizedSearchCV:
    """
    RandomizedSearchCV wrapper for Random Forest tuning. Week 7 Tutorial 2,
    Cell 39. Calling `.fit(X_train, y_train)` on the result reproduces the
    tuning procedure; `.best_estimator_` is the tuned model. Defaults
    reproduce the notebook exactly.
    """
    return RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=n_jobs),
        param_distributions=param_distributions or random_forest_search_space(),
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def build_gradient_boosting(
    max_depth: int = 6,
    learning_rate: float = 0.1,
    max_iter: int = 300,
    class_weight: str | None = "balanced",
    random_state: int = RANDOM_STATE,
) -> HistGradientBoostingClassifier:
    """
    Gradient Boosting. Week 7 Tutorial 2, Cell 43. Defaults reproduce the
    notebook exactly.
    """
    return HistGradientBoostingClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )


def build_mlp(
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    alpha: float = 1e-3,
    max_iter: int = 500,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """
    Small scaled MLP. Week 7 Tutorial 2, Cell 47. Defaults reproduce the
    notebook exactly.
    """
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state,
        ),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(y_true, y_pred, esi1_label: int = 1) -> dict:
    """
    Compute the four headline metrics used across Weeks 6-7:
    accuracy, macro-F1, weighted-F1, and ESI-1 recall.

    Direct translation of Week 6 Tutorial 3, Cells 21/26/31/33
    (`classification_report`, `f1_score(..., average="macro"/"weighted")`,
    `recall_score(..., labels=[1], average=None)`).

    Parameters
    ----------
    y_true : array-like
        True ESI labels for the test set.
    y_pred : array-like
        Model predictions for the same test set.
    esi1_label : int
        The label value representing ESI level 1 (default 1, as used
        throughout every notebook). Exposed as a parameter rather than
        hardcoded so this function is reusable if a future dataset
        encodes ESI differently -- current behaviour is unchanged.

    Returns
    -------
    dict with keys: accuracy, macro_f1, weighted_f1, recall_esi1
    """
    accuracy = (y_pred == y_true).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    recall_esi1 = recall_score(y_true, y_pred, labels=[esi1_label], average=None)[0]

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "recall_esi1": recall_esi1,
    }


# ---------------------------------------------------------------------------
# Config-driven dispatch (used by scripts/train.py)
# ---------------------------------------------------------------------------

# Maps the string used in config.yaml's final_model.type to the matching
# builder function above. This is what lets scripts/train.py pick a model
# by reading the config file, instead of a hardcoded if/elif chain.
MODEL_BUILDERS = {
    "DummyClassifier": build_dummy_baseline,
    "LogisticRegression": build_logistic_regression,
    "DecisionTreeClassifier": build_decision_tree,
    "RandomForestClassifier": build_random_forest,
    "GradientBoostingClassifier": build_gradient_boosting,
    "MLPClassifier": build_mlp,
}


def build_model_from_config(final_model_config: dict):
    """
    Build a model purely from a config dict shaped like config.yaml's
    `final_model` section, e.g.:

        {
            "type": "LogisticRegression",
            "hyperparameters": {"max_iter": 1000, "random_state": 42},
        }

    Raises a clear error if `type` isn't a known builder, rather than a
    confusing KeyError deep in a pipeline.
    """
    model_type = final_model_config["type"]
    hyperparameters = final_model_config.get("hyperparameters", {})

    if model_type not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model type '{model_type}' in config. "
            f"Known types: {sorted(MODEL_BUILDERS)}"
        )

    builder = MODEL_BUILDERS[model_type]
    return builder(**hyperparameters)
