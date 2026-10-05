# tausurv

Survival analysis for Python: nonparametric estimators, regression models,
tree ensembles and neural networks behind one interface, with support for competing
risks and minimal dependencies.

**Documentation:** https://scai-bio.github.io/tausurv/

## Installation

```bash
pip install tausurv
```

Requires Python 3.12 or newer. Optional extras:

| extra | adds |
|---|---|
| `tausurv[plot]` | plotting with matplotlib |
| `tausurv[boost]` | `SurvivalBoost`, via scikit-learn |
| `tausurv[tune]` | hyperparameter search and nested cross-validation, via Optuna |
| `tausurv[deepsurv]` | the four DeepSurv benchmark tables stored as HDF5 |

The neural models in `tausurv.nn` need PyTorch, which is not installed
automatically because the right build depends on your hardware. Install it
first by following https://pytorch.org/get-started/locally/.

## Example

```python
import tausurv as ts

X, time, event = ts.datasets.load_dataset("pbc")
time = time / 365.25  # days -> years

km = ts.nonparametric.kaplan_meier(time, event)
km(5.0)  # 5-year survival, about 0.70

covariates = X.select("age", "bili", "albumin").to_numpy()
cox = ts.linear.CoxPH().fit(covariates, time, event)
cox.predict_survival_function(covariates[:3], [1.0, 5.0, 10.0])
```

## What's included

- **Nonparametric:** Kaplan-Meier, Nelson-Aalen, Aalen-Johansen, censoring
  distribution.
- **Regression:** Cox proportional hazards, Fine-Gray, Weibull, log-normal and
  log-logistic AFT.
- **Trees:** survival tree, random survival forest, gradient-boosted
  `SurvivalBoost`.
- **Neural:** DeepSurv, DeepHit, DSM, logistic hazard, copula-based and
  HACSurv models.
- **Metrics:** Harrell's, Uno's and Antolini's C-index, Brier score and
  integrated Brier score, time-dependent AUC, calibration.
- **Model selection:** survival-aware splits, cross-validation,
  hyperparameter tuning and nested cross-validation.
- **Plotting:** survival and cumulative-incidence curves with at-risk tables,
  forest plots, calibration, metrics over time.
- **Datasets:** about 100 published cohorts that download on first use and
  are verified against a pinned checksum.

## Development

The repository is a uv workspace. Install PyTorch into it with the build for
your hardware, then sync:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv sync --all-packages --all-groups
uv run pytest
```

## License

MIT. Developed at Fraunhofer SCAI.
