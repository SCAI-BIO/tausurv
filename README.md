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

|  | tausurv | scikit-survival | lifelines | pycox | hazardous |
|---|:---:|:---:|:---:|:---:|:---:|
| **Nonparametric** |  |  |  |  |  |
| Kaplan-Meier | ✓ | ✓ | ✓ |  |  |
| Nelson-Aalen | ✓ | ✓ | ✓ |  |  |
| Aalen-Johansen | ✓ | ✓ | ✓ |  |  |
| Left truncation (delayed entry) |  | ✓ | ✓ |  |  |
| Log-rank test |  | ✓ | ✓ |  |  |
| Parametric univariate fits |  |  | ✓ |  |  |
| **Regression** |  |  |  |  |  |
| Cox proportional hazards | ✓ | ✓ | ✓ |  |  |
| Penalized Cox (L1, L2, elastic net) |  | ✓ | ✓ |  |  |
| Time-varying covariates |  |  | ✓ |  |  |
| Proportional-hazards test |  |  | ✓ |  |  |
| AFT (Weibull, log-normal, log-logistic) | ✓ |  | ✓ |  |  |
| Aalen additive hazards |  |  | ✓ |  |  |
| Fine-Gray | ✓ |  |  |  |  |
| **Trees and boosting** |  |  |  |  |  |
| Survival tree | ✓ | ✓ |  |  |  |
| Random survival forest | ✓ | ✓ |  |  |  |
| Gradient boosting | ✓ | ✓ |  |  | ✓ |
| Competing-risks gradient boosting | ✓ |  |  |  | ✓ |
| Survival support vector machine |  | ✓ |  |  |  |
| **Neural networks** |  |  |  |  |  |
| DeepSurv | ✓ |  |  | ✓ |  |
| DeepHit, with competing risks | ✓ |  |  | ✓ |  |
| Logistic hazard | ✓ |  |  | ✓ |  |
| Cox-Time, MTLR, PC-Hazard |  |  |  | ✓ |  |
| Deep survival machines | ✓ |  |  |  |  |
| CopulaSurv (dependent censoring) | ✓ |  |  |  |  |
| HACSurv (dependent competing risks) | ✓ |  |  |  |  |
| **Discrimination** |  |  |  |  |  |
| Harrell's concordance | ✓ | ✓ | ✓ |  |  |
| Uno's concordance (IPCW) | ✓ | ✓ |  |  |  |
| Antolini's time-dependent concordance | ✓ |  |  | ✓ |  |
| Time-dependent AUC | ✓ | ✓ |  |  |  |
| **Prediction error and calibration** |  |  |  |  |  |
| Brier score (IPCW) | ✓ | ✓ |  | ✓ | ✓ |
| Integrated Brier score | ✓ | ✓ |  | ✓ | ✓ |
| Censored negative log-likelihood | ✓ |  |  |  |  |
| CRPS | ✓ |  |  |  |  |
| Calibration curve | ✓ |  | ✓ |  | ✓ |
| D-calibration | ✓ |  |  |  |  |
| **Competing-risks metrics** |  |  |  |  |  |
| Cause-specific concordance | ✓ |  |  |  | ✓ |
| Cause-specific Brier score | ✓ |  |  |  | ✓ |
| Cause-specific AUC | ✓ |  |  |  |  |
| Cause-specific calibration | ✓ |  |  |  | ✓ |
| **Workflow** |  |  |  |  |  |
| Survival-aware cross-validation and tuning | ✓ | partly |  |  |  |
| Nested cross-validation | ✓ |  |  |  |  |
| Save and load fitted models | partly | ✓ | ✓ | ✓ | ✓ |
| Bundled datasets | ✓ | ✓ | ✓ | ✓ |  |
| Survival curves with at-risk tables | ✓ |  | ✓ |  |  |
| Time-dependent SHAP plots | ✓ |  |  |  |  |

Checked against scikit-survival 0.27.0, lifelines 0.30.3, pycox 0.3.0 and
hazardous 0.2.0. The [full comparison](https://scai-bio.github.io/tausurv/comparison/)
explains the partial cells.

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
