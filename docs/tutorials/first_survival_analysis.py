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
# # Your first survival analysis
#
# This tutorial walks through a complete first analysis on a real clinical-trial cohort: load the Mayo Clinic primary biliary cholangitis (PBC) trial, fit Kaplan-Meier curves, read off concrete numbers (median and horizon-specific survival), and see what visual comparisons look like for a covariate that does not matter (treatment arm) and one that matters a great deal (histologic stage).

# %% [markdown]
# ## Set up
#
# Throughout the docs we import `tausurv as ts` and access subpackages explicitly (`ts.datasets`, `ts.nonparametric`, `ts.plot`). The convention mirrors `numpy as np` and `pandas as pd` and keeps the library structure visible at the call site. A single `ts.plot.set_style("publication")` call configures matplotlib for the docs aesthetic; no manual `rcParams` are needed in user code.

# %%
import numpy as np

import tausurv as ts

# %config InlineBackend.figure_format = 'svg'

ts.plot.set_style("publication")

# %% [markdown]
# ## The PBC trial cohort
#
# The Mayo Clinic primary biliary cholangitis trial (1974-1984) followed 418 patients with PBC liver disease. The first 312 were randomised to D-penicillamine or placebo; the remaining 106 are non-randomised follow-up. The outcome is time to death, in days. Following the standard single-event recoding, liver transplants and end-of-study are treated as censoring.
#
# `ts.datasets.load_pbc()` returns a `SurvivalBunch` that tuple-unpacks as `(X, Y, delta)`. The bunch also carries metadata (`.name`, `.description`, `.citation`).

# %%
pbc = ts.datasets.load_pbc()
X, Y, delta = pbc

# Convert days to years for a more readable time axis.
Y = Y / 365.25

n = len(Y)
n_events = int(delta.sum())
print(f"{n} patients, {n_events} deaths ({n_events / n:.0%})")
print(f"median follow-up: {np.median(Y):.2f} years")
print(f"covariates ({X.width}): {', '.join(X.columns)}")

# %% [markdown]
# Beyond `(Y, delta)`, PBC carries the treatment arm (`trt`, 1 = D-penicillamine, 2 = placebo, null = non-randomised follow-up), demographics (`age`, `sex`), lab values measured at trial entry (`bili` for bilirubin, `albumin`, `chol`, `copper`, ...), and the histologic disease stage (`stage`, 1-4, higher is worse). We will use `trt` and `stage` directly in this tutorial.

# %% [markdown]
# ## Overall survival
#
# `ts.plot.km` on raw $(Y, \delta)$ arrays fits Kaplan-Meier internally and renders the publication treatment by default: a logit-transformed pointwise 95% confidence band (Borgan & Liestøl 1990) and an at-risk table aligned to the time axis (Pocock, Clayton & Altman, *Lancet* 2002).

# %%
ts.plot.km(Y, delta, xlabel="years from registration")

# %% [markdown]
# A visual reading is the right first step, but the curve also gives concrete numbers. The Kaplan-Meier estimate is a `StepFunction`; call it on any time to read $\hat S(t)$, and inspect its `.time` / `.value` arrays directly to find the median.

# %%
S = ts.nonparametric.kaplan_meier(Y, delta)

# Median: smallest time where the estimated survival has dropped to or below 0.5.
below_half = S.value <= 0.5
if below_half.any():
    median_t = float(S.time[np.argmax(below_half)])
    median_str = f"{median_t:.2f} years"
else:
    median_str = "not reached within follow-up"

print(f"median survival:  {median_str}")
print(f"5-year survival:  {float(S(5.0)):.1%}")
print(f"10-year survival: {float(S(10.0)):.1%}")

# %% [markdown]
# About three-quarters of the cohort is alive at five years, dropping to roughly 40% at ten. The median survival lies near the ten-year mark.

# %% [markdown]
# ## Does the treatment work?
#
# The trial's primary question was whether D-penicillamine extends survival relative to placebo. We restrict to the randomised 312 patients and split by arm.

# %%
arm_raw = X["trt"].to_numpy()
randomised = ~np.isnan(arm_raw)
arm = np.where(arm_raw[randomised] == 1, "D-penicillamine", "Placebo")

ts.plot.km(
    Y[randomised],
    delta[randomised],
    group=arm,
    xlabel="years from registration",
)

# %% [markdown]
# The two curves track each other inside their confidence bands. The same numerical horizons confirm the visual impression:

# %%
for label, code in [("D-penicillamine", 1), ("Placebo", 2)]:
    mask = arm_raw == code
    S_arm = ts.nonparametric.kaplan_meier(Y[mask], delta[mask])
    print(
        f"{label:18s}  5-year: {float(S_arm(5.0)):.1%}   "
        f"10-year: {float(S_arm(10.0)):.1%}   "
        f"(n = {int(mask.sum())}, events = {int(delta[mask].sum())})"
    )

# %% [markdown]
# Visually and numerically, the arms are within a percentage point or two of each other at every horizon. This matches the trial's published conclusion: D-penicillamine showed no survival benefit.
#
# Two things to pause on. First, the absence of visible separation in a trial of this size does not prove the absence of any effect; it limits how large an effect could plausibly be. Second, this is what a true null result looks like in a survival study. Most negative trials look more like this than like dramatically diverging curves.

# %% [markdown]
# ## A covariate that does matter: histologic stage
#
# Treatment was a randomised covariate that turned out not to matter for this disease. Histologic stage at baseline is a non-randomised covariate that matters a great deal. Stage is the pathologist's reading of biopsy material, scored 1 (mild fibrosis) through 4 (cirrhosis).

# %%
stage_raw = X["stage"].to_numpy()
has_stage = ~np.isnan(stage_raw)
stage = stage_raw[has_stage].astype(int)

ts.plot.km(
    Y[has_stage],
    delta[has_stage],
    group=np.array([f"stage {k}" for k in stage]),
    xlabel="years from registration",
)

# %% [markdown]
# The four curves are visibly distinct, and the confidence bands stay clearly apart for most of follow-up:

# %%
for k in range(1, 5):
    mask = stage_raw == k
    S_k = ts.nonparametric.kaplan_meier(Y[mask], delta[mask])
    print(
        f"stage {k}  5-year: {float(S_k(5.0)):.1%}   "
        f"10-year: {float(S_k(10.0)):.1%}   "
        f"(n = {int(mask.sum())}, events = {int(delta[mask].sum())})"
    )

# %% [markdown]
# Stage 1 patients are still mostly alive at ten years; stage 4 patients are mostly dead by year five. The contrast with the treatment-arm comparison is the lesson: visible separation is what a real prognostic effect looks like.

# %% [markdown]
# ## What Kaplan-Meier does and does not give you
#
# Kaplan-Meier is a visual and descriptive tool. It tells you the marginal survival in a defined population, optionally split by one categorical covariate at a time. It does not tell you:
#
# - The magnitude of an effect adjusted for other covariates. The stage effect above is unadjusted; some of the visible separation may be confounded by age or baseline lab values.
# - A single-number summary of the contrast that you can put in a paper or a clinical-decision aid.
# - How to predict survival for a new patient given their covariates.
#
# The next tutorial fits a Cox proportional-hazards model to this cohort, adjusts for multiple covariates simultaneously, and quantifies the stage effect with a hazard ratio.

# %% [markdown]
# ## References
#
# - Therneau, T. & Grambsch, P. (2000). *Modeling Survival Data: Extending the Cox Model.* Springer. (Canonical worked source for the PBC dataset.)
# - Borgan, Ø. & Liestøl, K. (1990). A note on confidence intervals and bands for the survival function based on transformations. *Scandinavian Journal of Statistics* 17.
# - Pocock, S. J., Clayton, T. C. & Altman, D. G. (2002). Survival plots of time-to-event outcomes in clinical trials. *Lancet* 359.
