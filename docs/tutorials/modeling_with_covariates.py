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
# # Modeling risk with covariates
#
# In the previous tutorial Kaplan-Meier curves gave us a visual reading of survival in the PBC cohort, but they could only handle one categorical covariate at a time and gave no single-number summary of effect size. This tutorial fits a Cox proportional-hazards model on the same cohort. The Cox model adjusts for several covariates simultaneously, returns a hazard ratio per covariate, and predicts a full survival curve for any patient profile. We end by evaluating the fit on a held-out test split with two complementary checks: calibration at a fixed horizon and the Brier score over time.

# %% [markdown]
# ## Set up

# %%
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import tausurv as ts
from tausurv.step import StepFunction

# %config InlineBackend.figure_format = 'svg'

ts.plot.set_style("publication")

# %% [markdown]
# ## Looking at the cohort with polars
#
# `ts.datasets.load_pbc` returns a `SurvivalBunch` that tuple-unpacks as `(X, Y, delta)`. The covariates live in a polars `DataFrame`, which has its own inspection idioms. We use them rather than printing summary statistics by hand.

# %%
pbc = ts.datasets.load_pbc()
X, Y, delta = pbc
Y = Y / 365.25

X.head()

# %% [markdown]
# Seventeen covariates, mostly numeric, a few categorical (`sex`, `edema`). Some columns have missing values; `null_count` reports the per-column count.

# %%
X.null_count()

# %% [markdown]
# Treatment arm (`trt`) is missing for the 106 non-randomised follow-up patients. Several lab measurements (`chol`, `copper`, `alk.phos`, `ast`, `trig`, `platelet`, `protime`) are missing for a handful of patients. We will pick five covariates with clinical importance and low missingness: **age**, **sex**, **stage** (histologic stage), **bili** (bilirubin) and **albumin**.

# %%
keep = ["age", "sex", "stage", "bili", "albumin"]
X.select(keep).describe()

# %% [markdown]
# ## Preparing the covariates
#
# `polars` carries us most of the way: select the kept columns with their row index, drop rows where any are missing, encode `sex` as a `0/1` indicator, and convert to NumPy at the boundary. Tracking the row index lets us subset `Y` and `delta` to match.

# %%
df = pbc.X.select(keep).with_row_index("_row").drop_nulls()
kept_rows = df["_row"].to_numpy()
df = (
    df.drop("_row")
    .with_columns(
        sex_male=(pl.col("sex") == "m").cast(pl.Int8),
    )
    .drop("sex")
    .select(["age", "stage", "bili", "albumin", "sex_male"])
)

X_array = df.to_numpy().astype(np.float64)
Y_kept = Y[kept_rows]
delta_kept = delta[kept_rows]
feature_names = list(df.columns)

print(f"{len(Y_kept)} patients kept ({len(Y) - len(Y_kept)} dropped for missingness)")

# %% [markdown]
# A 70/30 train/test split lets us evaluate the fitted model on patients it has not seen. PBC is small, so the absolute numbers are modest; the goal is to keep the evaluation honest rather than to claim a definitive benchmark.

# %%
rng = np.random.default_rng(seed=42)
order = rng.permutation(len(Y_kept))
split = int(0.7 * len(Y_kept))
train, test = order[:split], order[split:]

X_train, Y_train, d_train = X_array[train], Y_kept[train], delta_kept[train]
X_test, Y_test, d_test = X_array[test], Y_kept[test], delta_kept[test]

print(
    f"train: n = {len(Y_train)}, events = {int(d_train.sum())}    "
    f"test: n = {len(Y_test)}, events = {int(d_test.sum())}"
)

# %% [markdown]
# ## Fit the Cox model
#
# The Cox model parameterises the hazard as $\lambda(t \mid x) = \lambda_0(t) \exp(\beta^\top x)$ and estimates $\beta$ by maximising Cox's partial likelihood. `ts.linear.CoxPH` takes the design matrix and the observed-time / event-indicator arrays and returns a fitted estimator carrying `coef_` (the $\hat\beta$) and a baseline cumulative hazard.

# %%
cox = ts.linear.CoxPH().fit(X_train, Y_train, d_train)

# %% [markdown]
# ## Reading the coefficients
#
# Hazard ratios ($\exp(\hat\beta)$) are more interpretable than log-hazards. A polars table is a clean way to display them; the forest plot is the visual counterpart.

# %%
coefs = pl.DataFrame(
    {
        "covariate": feature_names,
        "coef": cox.coef_,
        "HR": np.exp(cox.coef_),
    }
).sort("HR", descending=True)
coefs

# %%
ts.plot.forest(
    names=feature_names,
    estimates=np.exp(cox.coef_),
    ci=np.exp(cox.confidence_intervals()),
    xlabel="Hazard ratio (per 1-unit increase)",
)

# %% [markdown]
# Reading the strongest effects, every additional unit of **bilirubin** (mg/dL) multiplies the hazard of death by roughly the bilirubin HR shown, holding the other covariates fixed; every step up in **histologic stage** does the same on its own scale; every additional year of **age** is a small multiplier. **Albumin** sits on the other side of one — higher serum albumin is associated with lower mortality, as a marker of preserved liver synthetic function.
#
# The bars are Wald 95% intervals from the inverse observed information at $\hat\beta$, exponentiated onto the hazard-ratio scale.

# %% [markdown]
# ## What the model says about specific patients
#
# Once $\beta$ and $\hat \Lambda_0$ are known the model can produce a full survival curve for any covariate profile through $\hat S(t \mid x) = \exp(-\hat\Lambda_0(t) \exp(\hat\beta^\top x))$. We contrast two illustrative profiles: an early-stage patient with normal labs and a late-stage patient with markedly abnormal labs.

# %%
profiles = {
    "Early PBC, age 50, normal labs": np.array(
        [50.0, 1.0, 0.6, 4.0, 0.0]  # age, stage, bili, albumin, sex_male
    ),
    "Advanced PBC, age 70, abnormal labs": np.array([70.0, 4.0, 8.0, 2.8, 0.0]),
}

times_grid = np.linspace(0.1, 12.0, 240)
S_profiles = cox.predict_survival_function(
    np.stack(list(profiles.values())), times_grid
)

fig, ax = plt.subplots()
for label, s in zip(profiles, S_profiles, strict=True):
    S_obj = StepFunction(time=times_grid, value=s, baseline=1.0)
    ts.plot.km(S_obj, ax=ax, label=label)
ax.set_xlabel("years from registration")

# %% [markdown]
# The early-stage profile sits high throughout follow-up; the late-stage profile drops below 0.5 inside a few years. This is the model's projection for two hypothetical patients with identical clinical context except for the five covariates the model was fitted on.

# %% [markdown]
# ## Why is the advanced profile so high-risk?
#
# A hazard ratio tells us how much each feature multiplies the hazard, but it does not say which features drive the gap between a *particular* patient's predicted survival and the cohort average. **SurvSHAP(t)** (Krzyzinski et al. 2023) decomposes that gap into per-feature contributions at every time point. For Cox with five features, the exact Shapley values are cheap to compute by brute force over feature subsets.

# %%
from itertools import combinations, product
from math import factorial

times_shap = np.linspace(0.1, 12.0, 100)
x_ref = X_train.mean(axis=0)
x_target = profiles["Advanced PBC, age 70, abnormal labs"]
d = len(x_target)

# Precompute $S(t \mid x)$ for every one of the $2^d = 32$ feature-subset
# assignments in a single batched call. ``masks[i, j] = 1`` means feature
# ``j`` takes the target's value in row ``i``; ``0`` means it takes the
# cohort-mean reference.
masks = np.array(list(product([0, 1], repeat=d)), dtype=bool)
X_subsets = np.where(masks, x_target, x_ref)
S_subsets = cox.predict_survival_function(X_subsets, times_shap)
mask_row = {frozenset(np.where(m)[0]): i for i, m in enumerate(masks)}

S_baseline = S_subsets[mask_row[frozenset()]]

shap_vals = np.zeros((d, len(times_shap)))
for j in range(d):
    others = [k for k in range(d) if k != j]
    for k in range(d):
        for subset in combinations(others, k):
            weight = factorial(k) * factorial(d - k - 1) / factorial(d)
            shap_vals[j] += weight * (
                S_subsets[mask_row[frozenset(list(subset) + [j])]]
                - S_subsets[mask_row[frozenset(subset)]]
            )

ts.plot.shap.local_decomposition(
    shap_vals,
    baseline_survival=S_baseline,
    times=times_shap,
    features=feature_names,
    xlabel="years from registration",
)

# %% [markdown]
# Two panels share a time axis. The top panel shows the dashed baseline survival (a hypothetical patient with the cohort-mean covariates) and the solid prediction for the advanced-PBC profile; the vertical gap between them *is* the total SHAP effect on survival. The bottom panel decomposes that gap into per-feature contributions: positive contributions stack upward from zero, negative downward.
#
# Stage and bilirubin dominate the downward shift, and they do so most strongly at the intermediate horizons where the baseline survival has the most room to fall. Age contributes a smaller decrement; albumin (low at 2.8 vs the cohort mean) and sex play minor roles. For Cox the Shapley loop is exact and fast; for larger feature sets or nonlinear models you would precompute SHAP arrays with an explainer of your choice (`shap`, `survshap`, ...) and pass them to the same plot.

# %% [markdown]
# ## Across the cohort: which features matter when?
#
# The local decomposition explains one patient. To see whether that pattern generalises — which features dominate across the training cohort and at which horizons — we compute SHAP for every training subject and aggregate. The Shapley sum is the same as above, but vectorised: precompute the survival prediction under each of the $2^d = 32$ feature-subset configurations once for the full cohort, then assemble per-feature contributions from those matrices.

# %%
subset_S = {}
for k in range(d + 1):
    for subset in combinations(range(d), k):
        X_mod = X_train.copy()
        not_in = [j for j in range(d) if j not in subset]
        X_mod[:, not_in] = x_ref[not_in]
        subset_S[frozenset(subset)] = cox.predict_survival_function(X_mod, times_shap)

shap_cohort = np.zeros((len(X_train), d, len(times_shap)))
for j in range(d):
    others = [m for m in range(d) if m != j]
    for k in range(d):
        for subset in combinations(others, k):
            weight = factorial(k) * factorial(d - k - 1) / factorial(d)
            shap_cohort[:, j, :] += weight * (
                subset_S[frozenset(list(subset) + [j])] - subset_S[frozenset(subset)]
            )

ts.plot.shap.feature_time_heatmap(
    shap_cohort,
    times=times_shap,
    features=feature_names,
    xlabel="years from registration",
)

# %% [markdown]
# Rows are features, columns are time, cell brightness is the mean absolute SHAP across training subjects at that (feature, time). Bilirubin and stage dominate the cohort-wide picture, with attribution peaking in the 2-5 year range where the baseline survival curve has the most room to move. Age, albumin and sex play more modest roles, and their attribution is more uniform across time. The pattern confirms that the local decomposition above was not idiosyncratic to one patient: the same two features drive most of the variation in predicted survival across the cohort.

# %% [markdown]
# ## How well does the model fit?
#
# Two complementary checks on the held-out test set. **Calibration** at a fixed horizon asks whether predicted survival probabilities are right on average within bins of predicted risk. **Brier over time** scores discrimination plus calibration jointly at each horizon, against a perfect-prediction baseline of zero.

# %%
eval_grid = np.linspace(0.1, 12.0, 200)
S_test = cox.predict_survival_function(X_test, eval_grid)

predicted_5y, observed_5y, _ = ts.metrics.calibration.curve(
    Y_test,
    d_test,
    S_test,
    eval_grid,
    t=5.0,
    n_bins=5,
)

ts.plot.calibration(
    predicted_5y,
    observed_5y,
    xlabel="Predicted 5-year survival",
    ylabel="Observed 5-year survival (KM in bin)",
)

# %% [markdown]
# Each point is one of five quantile bins of predicted five-year survival; its x-coordinate is the bin's mean prediction and its y-coordinate is the Kaplan-Meier estimate of survival at five years among the bin's test patients. A perfectly calibrated model would have all points on the dashed identity line.

# %% [markdown]
# A single test split gives one point estimate of the Brier score at each horizon. With a cohort this size, that estimate is noisy. **5-fold cross-validation** refits the model on each held-out 80% and scores the remaining 20%, giving five estimates per horizon. Passing the resulting `(n_folds, n_horizons)` matrix to `ts.plot.brier_over_time` as `values` draws the mean as the line and a $\pm 1$ standard-deviation band around it.

# %%
horizons = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
n_folds = 5

rng_cv = np.random.default_rng(seed=42)
fold_indices = np.array_split(rng_cv.permutation(len(Y_kept)), n_folds)
brier_folds = np.zeros((n_folds, len(horizons)))

for i, te in enumerate(fold_indices):
    tr = np.setdiff1d(np.arange(len(Y_kept)), te)
    cox_cv = ts.linear.CoxPH().fit(X_array[tr], Y_kept[tr], delta_kept[tr])
    S_cv = cox_cv.predict_survival_function(X_array[te], horizons)
    brier_folds[i] = ts.metrics.brier.score(Y_kept[te], delta_kept[te], S_cv, horizons)

ts.plot.brier_over_time(
    horizons,
    brier_folds,
    xlabel="years from registration",
)

# %% [markdown]
# Lower is better. The Brier score combines discrimination (does the model rank patients correctly) and calibration (are the absolute probabilities right) into one number per horizon; integrated over a window it becomes the Integrated Brier Score (IBS), available as `ts.metrics.brier.integrated`. The band's width is the cross-fold standard deviation; it widens at later horizons where fewer patients remain at risk and the per-fold estimates diverge more.

# %% [markdown]
# ## Caveats and where to go from here
#
# What this tutorial demonstrates and what it deliberately does not.
#
# - **Proportional hazards.** The Cox model assumes the hazard ratio for each covariate is constant over time. PBC arguably violates this for bilirubin and stage at long follow-up. A diagnostic tutorial on Schoenfeld residuals comes later.
# - **Competing risks.** Liver transplant was treated as censoring in the single-event recoding. That is the standard simplification, but it conflates "we stopped watching" with "the patient received a transplant, which changes their prognosis." The competing-risks tutorial revisits this using `ts.linear.FineGray` and the cause-specific cumulative incidence.
# - **Honest evaluation.** The 70/30 split here is the simplest possible. A real reporting pipeline uses repeated cross-validation, bootstraps the calibration curves, and reports IBS / time-dependent AUC alongside Brier. The evaluation tutorial covers each.
# - **Causal interpretation.** A hazard ratio is a population summary, not a causal effect of changing a covariate. The separate `causurv` package covers what additional assumptions are needed before HRs become causal contrasts and why RMST differences are often a cleaner target.

# %% [markdown]
# ## References
#
# - Therneau, T. & Grambsch, P. (2000). *Modeling Survival Data: Extending the Cox Model.* Springer.
# - Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society, Series B* 34(2).
# - Breslow, N. (1974). Covariance analysis of censored survival data. *Biometrics* 30(1).
# - Graf, E., Schmoor, C., Sauerbrei, W. & Schumacher, M. (1999). Assessment and comparison of prognostic classification schemes for survival data. *Statistics in Medicine* 18.
