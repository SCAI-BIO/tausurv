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
# # Competing risks
#
# The two previous tutorials worked on PBC under a single-event recoding: deaths were the event of interest and liver transplants were folded into censoring. That simplification is convenient but it conflates "we stopped watching" with "the patient received a transplant" — two very different things. This tutorial recovers the three-level outcome (censored / transplant / death), shows visually why naïvely applying Kaplan-Meier to one cause overstates the incidence of that cause, fits a Fine-Gray model for the subdistribution hazard of death, and predicts cause-specific cumulative incidence curves for the same two patient profiles used in the Cox tutorial.

# %% [markdown]
# ## Set up

# %%
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import tausurv as ts

# %config InlineBackend.figure_format = 'svg'

ts.plot.set_style("publication")

# %% [markdown]
# ## Bringing the competing event back
#
# `ts.datasets.load_pbc()` returns the single-event bunch. The `pbc:transplant` variant reads the same 418 patients as competing risks, coded with the convention used throughout the library: cause 0 is censored, cause 1 is the event of interest (death), cause 2 is the competing event (transplant).

# %%
pbc = ts.datasets.load_dataset("pbc:transplant")
X_full, Y_full, _ = pbc
Y_full = Y_full / 365.25  # days -> years
event = pbc.cause

counts = pl.DataFrame(
    {
        "outcome": ["censored", "death", "transplant"],
        "n": [
            int((event == 0).sum()),
            int((event == 1).sum()),
            int((event == 2).sum()),
        ],
    }
)
counts

# %% [markdown]
# About a quarter of the cohort received a transplant. In the single-event recoding those patients were merged with the truly-censored. They are not censored — we know exactly what happened to them — and they cannot subsequently die without a graft.

# %% [markdown]
# ## The pitfall: $1 - \widehat{\mathrm{KM}}$ overstates the cause incidence
#
# Naïve practice is to compute Kaplan-Meier on "death or not" — equivalent to treating transplant as censoring — then report $1 - \hat S$ as the incidence of death. This is biased upward whenever the competing event has nonzero rate, because subjects removed from the at-risk set as "censored transplants" are implicitly assumed to have the same death hazard as the rest of the cohort going forward, which is false (a graft fundamentally changes their trajectory).
#
# The Aalen-Johansen estimator handles competing causes correctly: $\hat F_k(t)$ is the probability of failing from cause $k$ by time $t$ in a world where the competing cause keeps happening.

# %%
death_or_not = (event == 1).astype(np.int8)
S_naive = ts.nonparametric.kaplan_meier(Y_full, death_or_not)
F_aj = ts.nonparametric.aalen_johansen(Y_full, event, cause=1)

t_grid = np.linspace(0.1, 12.0, 240)
naive_cif = 1.0 - S_naive(t_grid)
aj_cif = F_aj(t_grid)

fig, ax = plt.subplots()
ax.step(
    t_grid,
    naive_cif,
    where="post",
    label=r"$1 - \widehat{\mathrm{KM}}$ (transplant as censoring)",
)
ax.step(t_grid, aj_cif, where="post", label="Aalen-Johansen $\\hat F_\\mathrm{death}$")
ax.set_xlabel("years from registration")
ax.set_ylabel("cumulative incidence of death")
ax.set_xlim(0, 12)
ax.set_ylim(0, None)
ax.legend(loc="lower right")

# %% [markdown]
# The naïve curve sits above the Aalen-Johansen curve at every horizon and the gap widens with time. At ten years the naïve estimate overstates the cumulative incidence of death by several percentage points. In real datasets with higher competing-event rates the gap can be twenty percentage points or more (Andersen et al. 2012 review the bias on transplant and oncology cohorts).

# %% [markdown]
# ## Both causes at once
#
# `ts.plot.cif` renders the cause-specific cumulative incidence curves for every cause it sees in the indicator. It puts the curves on a stacked-to-one display and adds an at-risk table aligned to the time axis, matching the conventions used for Kaplan-Meier.

# %%
ts.plot.cif(
    Y_full,
    event,
    cause_labels={1: "death", 2: "transplant"},
    xlabel="years from registration",
)

# %% [markdown]
# Death dominates the incidence early; transplant incidence rises more steadily and accounts for roughly a quarter of the cohort by the end of follow-up. The remaining height between the top curve and one is the probability of still being event-free.

# %% [markdown]
# ## Fitting a Fine-Gray model for death
#
# The Fine-Gray model parameterises the **subdistribution hazard** of a chosen cause:
# $$
# \lambda^{\mathrm{sub}}_k(t \mid x) = \lambda^{\mathrm{sub}}_{k,0}(t) \exp(\beta^\top x).
# $$
# Subjects who experience a competing event stay in the risk set with IPCW-decaying weight rather than dropping out (which is what Cox would do). The result is a model on the cumulative incidence scale: a positive coefficient pushes the cumulative incidence curve *up*, a negative one pushes it down. We use the same five covariates and the same train/test split as the previous tutorial.

# %%
keep = ["age", "sex", "stage", "bili", "albumin"]
df = X_full.select(keep).with_row_index("_row").drop_nulls()
kept = df["_row"].to_numpy()
df = (
    df.drop("_row")
    .with_columns(sex_male=(pl.col("sex") == "m").cast(pl.Int8))
    .drop("sex")
    .select(["age", "stage", "bili", "albumin", "sex_male"])
)
X_array = df.to_numpy().astype(np.float64)
event_kept = event[kept]
Y_kept = Y_full[kept]
feature_names = list(df.columns)

rng = np.random.default_rng(seed=42)
order = rng.permutation(len(Y_kept))
split = int(0.7 * len(Y_kept))
train, test = order[:split], order[split:]

fg = ts.linear.FineGray(cause=1).fit(X_array[train], Y_kept[train], event_kept[train])

print(
    f"train: n = {len(train)}, deaths = {int((event_kept[train] == 1).sum())}, "
    f"transplants = {int((event_kept[train] == 2).sum())}"
)

# %% [markdown]
# Subdistribution hazard ratios are $\exp(\hat\beta)$ on the cumulative-incidence scale, *not* the cause-specific hazard scale. A SHR of 1.5 for stage means stage shifts the **incidence curve** for death by a 1.5-fold multiplier in the proportional-subdistribution-hazards sense, not that the instantaneous death rate among survivors is 1.5x higher.

# %%
ts.plot.forest(
    names=feature_names,
    estimates=np.exp(fg.coef_),
    xlabel="Subdistribution hazard ratio (Fine-Gray)",
)

# %% [markdown]
# Reading the SHRs: stage and bilirubin pull the death CIF up the hardest, mirroring what the Cox tutorial found for cause-specific hazards. The magnitudes differ slightly because the models answer different questions — Austin & Fine (2017) reviewed 55 papers using Fine-Gray models and found only 9% interpreted the SHRs correctly. The most common mistake is reading a SHR as a cause-specific hazard ratio; they are different objects.

# %% [markdown]
# ## Predicting cumulative incidence for two patients
#
# `predict_cif` returns $\hat F_\mathrm{death}(t \mid x)$ on the same time grid we have been using. We reuse the two profiles from the Cox tutorial — an early-stage and a late-stage patient — and plot their predicted cumulative-incidence curves.

# %%
profiles = {
    "Early PBC, age 50, normal labs": np.array([50.0, 1.0, 0.6, 4.0, 0.0]),
    "Advanced PBC, age 70, abnormal labs": np.array([70.0, 4.0, 8.0, 2.8, 0.0]),
}
times_pred = np.linspace(0.1, 12.0, 240)
F_pred = fg.predict_cif(np.stack(list(profiles.values())), times_pred)

fig, ax = plt.subplots()
for label, f in zip(profiles, F_pred, strict=True):
    ax.step(times_pred, f, where="post", label=label)
ax.set_xlabel("years from registration")
ax.set_ylabel(r"$\hat F_\mathrm{death}(t \mid x)$")
ax.set_xlim(0, 12)
ax.set_ylim(0, 1.0)
ax.legend(loc="lower right")

# %% [markdown]
# The advanced profile reaches a cumulative incidence of death around 80% by year ten, while the early profile stays below 10%. These are predictions on the CIF scale — directly comparable to "what percentage of patients like this will die from PBC by year $t$" — rather than on a hazard scale that requires further integration to interpret.

# %% [markdown]
# ## How well does the model fit?
#
# Cause-specific Brier and concordance let us evaluate the model on the cumulative-incidence scale. `ts.metrics.brier.score_cause_specific` and `ts.metrics.concordance.harrell_cause_specific` are the competing-risks counterparts of the single-event metrics used in the Cox tutorial.

# %%
horizons = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
F_test = fg.predict_cif(X_array[test], horizons)

brier_cs = ts.metrics.brier.score_cause_specific(
    Y_kept[test], event_kept[test], F_test, horizons, cause=1
)

ts.plot.brier_over_time(
    horizons,
    brier_cs,
    xlabel="years from registration",
    ylabel="Brier score (cause-specific, death)",
)

# %% [markdown]
# Lower is better. The cause-specific Brier compares the predicted cumulative incidence at each horizon to the observed cause-specific incidence, with IPCW weighting that accounts for both censoring and competing events.

# %% [markdown]
# ## Caveats and where to go next
#
# - **SHR is not HR.** Austin & Fine (2017) is the canonical reference on the misinterpretation. A SHR is a multiplier on the cumulative-incidence curve under proportional subdistribution hazards; it is not the cause-specific hazard ratio. Always state which scale you are reporting.
# - **Cause-specific Cox is the other principled choice.** If the question is "what makes patients die at a higher rate while still at risk", cause-specific Cox (transplant treated as censoring within the *modelling* sense, with explicit acknowledgement) is the right tool. Latouche et al. (2013) recommend reporting both cause-specific hazards and cumulative-incidence functions in every competing-risks analysis.
# - **Multiple Fine-Gray fits can yield inconsistent CIFs.** Independently fitted Fine-Gray models for two or more causes can produce predicted cumulative incidences that sum to more than 1 (Austin, Putter, Lee & Steyerberg 2021). For prediction across causes, prefer the cause-specific Cox approach.
# - **Treatment is observational here.** PBC was randomised, but our adjusted models are not estimating a causal effect of treatment on death. The separate `causurv` package covers what assumptions are needed before HRs or SHRs become causal contrasts and why RMST contrasts are often a cleaner target.

# %% [markdown]
# ## References
#
# - Fine, J. P. & Gray, R. J. (1999). A proportional hazards model for the subdistribution of a competing risk. *JASA* 94(446).
# - Andersen, P. K., Geskus, R. B., de Witte, T. & Putter, H. (2012). Competing risks in epidemiology: possibilities and pitfalls. *International Journal of Epidemiology* 41.
# - Latouche, A., Allignol, A., Beyersmann, J., Labopin, M. & Fine, J. P. (2013). A competing risks analysis should report results on all cause-specific hazards and cumulative incidence functions. *Journal of Clinical Epidemiology* 66.
# - Austin, P. C. & Fine, J. P. (2017). Practical recommendations for reporting Fine-Gray model analyses for competing risk data. *Statistics in Medicine* 36.
# - Austin, P. C., Putter, H., Lee, D. S. & Steyerberg, E. W. (2021). Estimation of the absolute risk of cardiovascular disease in the presence of competing risks. *Statistics in Medicine* 40.
