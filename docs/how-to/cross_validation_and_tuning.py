# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: tausurv docs
#     language: python
#     name: tausurv-docs
# ---

# %% [markdown]
# # Cross-validate and tune a model
#
# Estimate how well a model generalises, pick its hyperparameters, and get an honest score for the tuned model, with `ts.model_selection`.

# %% [markdown]
# Three functions cover the workflow, each answering a different question:
#
# | Function | Question it answers |
# |---|---|
# | `cross_validate` | How well does this model, with these settings, do on unseen patients? |
# | `tune` | Which settings work best? |
# | `nested_cv` | How well does the *tuned* model do on unseen patients? |
#
# All three take the same two arguments: a `build` function that returns a fresh, unfitted model, and the data. They use stratified folds, so every fold gets its share of events, and the IPCW scorers estimate the censoring distribution on the training fold only. `tune` and `nested_cv` need Optuna: `pip install 'tausurv[tune]'`.

# %%
import numpy as np
import polars as pl

import tausurv as ts
from tausurv.model_selection import cross_validate, nested_cv, scoring, tune

# %% [markdown]
# We use the PBC cohort with the five covariates from [Tutorial 2](/tutorials/modeling-with-covariates/), times in years.

# %%
pbc = ts.datasets.load_pbc()
df = (
    pbc.X.with_columns(
        sex_male=(pl.col("sex") == "m").cast(pl.Int8),
        event_time=pl.Series(pbc.event_time / 365.25),
        event_indicator=pl.Series(pbc.event_indicator),
    )
    .select(
        "age", "stage", "bili", "albumin", "sex_male", "event_time", "event_indicator"
    )
    .drop_nulls()
)
X = df.drop("event_time", "event_indicator").to_numpy().astype(np.float64)
Y = df["event_time"].to_numpy()
E = df["event_indicator"].to_numpy()
X.shape

# %% [markdown]
# ## Cross-validation
#
# `cross_validate` fits a fresh model on each training fold and scores it on the held-out fold. Pass several scorers as a dict to get one column each. Here we use Uno's C-index up to five years (higher is better) and the integrated Brier score over years 1 to 5 (lower is better).

# %%
horizons = np.linspace(1.0, 5.0, 20)
scorers = {
    "uno_c": scoring.uno(horizon=5.0),
    "ibs": scoring.integrated_brier(horizons),
}


def build():
    return ts.trees.RandomSurvivalForest(n_estimators=100, seed=0)


cv = cross_validate(build, X, Y, E, scoring=scorers, cv=5, progress=False)
cv.scores

# %%
cv.scores.select(pl.col("uno_c", "ibs").mean())

# %% [markdown]
# The result also works as a predictor. On the study's own data it predicts *out of fold*: each patient is predicted by the model that did not see them. For new patients, `cv.ensemble` averages the fold models.

# %%
oof_risk = cv.predict(X)
new_risk = cv.ensemble.predict(X[:3])

# %% [markdown]
# ## Tuning
#
# To tune, give `build` a `trial` argument and draw each hyperparameter where it is used, with Optuna's `trial.suggest_*`. `tune` runs a Bayesian search in which each trial is scored by cross-validation, and by default refits the best settings on all the data.


# %%
def build_tuned(trial):
    return ts.trees.RandomSurvivalForest(
        n_estimators=100,
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 50, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 12),
        seed=0,
    )


best = tune(
    build_tuned,
    X,
    Y,
    E,
    scoring=scoring.uno(horizon=5.0),
    cv=3,
    n_trials=15,
    progress=False,
)
best.params, round(best.score, 3)

# %% [markdown]
# `best.model` is the refitted model, and `best.trials` has every trial with its parameters and score.

# %%
best.trials.sort("value", descending=True).head(5)

# %% [markdown]
# ## Nested cross-validation
#
# `best.score` is optimistic: the search picked the settings that happened to score highest on these folds, so it has partly fitted the folds' noise. To estimate how well the *whole procedure*, search included, generalises, `nested_cv` reruns the search inside every outer training fold and scores the winner on the outer test fold, which the search never saw.

# %%
ncv = nested_cv(
    build_tuned,
    X,
    Y,
    E,
    scoring=scoring.uno(horizon=5.0),
    outer=5,
    inner=3,
    n_trials=15,
    progress=False,
)
ncv.scores

# %%
ncv.scores["uno_c"].mean()

# %% [markdown]
# Report this mean, not `best.score`, as the tuned model's performance. `ncv.params` shows the settings each outer fold chose; when they vary a lot, the score is flat across that range and the exact value matters little.

# %%
pl.DataFrame(ncv.params)

# %% [markdown]
# ## Which one to use
#
# - **Comparing fixed models**: use `cross_validate`.
# - **Fitting a final model**: use `tune`, and ship `best.model`.
# - **Reporting the tuned model's performance**: use `nested_cv`, and report its outer scores.
#
# A nested study is `outer * (n_trials * inner + 1)` fits, 230 here, so keep `n_trials` small while you iterate. Pass `storage="sqlite:///study.db"` to `tune` to make a long search resumable, and use `.save(path)` on any result to keep the fitted models.
