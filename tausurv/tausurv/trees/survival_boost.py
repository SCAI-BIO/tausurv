from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tausurv.nonparametric.censoring_distribution import censoring_distribution
from tausurv.predictor import CompetingRisksPredictor

if TYPE_CHECKING:
    from sklearn.ensemble import HistGradientBoostingClassifier

# Floor for estimated censoring / survival probabilities before inversion,
# capping IPCW weights at 20. Matches the reference implementation.
_MIN_IPCW_PROB = 0.05


class SurvivalBoost(CompetingRisksPredictor):
    r"""Gradient-boosted competing-risks model (Alberge et al., 2025).

    Estimates the cause-specific cumulative incidence $\hat F_j(t \mid x)$
    for every cause and the event-free survival $\hat S(t \mid x)$ jointly,
    by boosting a multiclass classifier over randomly sampled time
    horizons. Each boosting iteration draws fresh horizons
    $\zeta_i \sim \mathcal U(0, t_{\max})$, stacks them onto the features,
    and trains one tree on the multiclass target

    - cause $j$, if the event of cause $j$ occurred by $\zeta_i$,
    - $0$ (event-free), if the subject is still under observation at
      $\zeta_i$,

    weighted by inverse-probability-of-censoring weights
    $1/\hat G(\cdot \mid x)$; subjects censored before their horizon get
    weight $0$. This loss is a strictly proper scoring rule for
    $(F_1, \dots, F_J, S)$, so the classifier's probabilities estimate the
    incidence functions directly.

    The trees are scikit-learn ``HistGradientBoostingClassifier``
    iterations, an optional dependency: install with
    ``pip install tausurv[boost]``. The import is deferred to
    :meth:`fit`, so importing ``tausurv`` does not require it.

    Parameters
    ----------
    hard_zero_fraction : float, default 0.1
        Fraction of sampled horizons pinned at $t = 0$, anchoring the
        model on $F_j(0) = 0$.
    n_iter : int, default 100
        Boosting iterations.
    learning_rate : float, default 0.05
        Shrinkage applied to each boosting iteration.
    max_leaf_nodes : int, default 31
        Leaves per base tree.
    max_depth : int, optional
        Depth cap per base tree. Unbounded by default; ``max_leaf_nodes``
        does the limiting.
    min_samples_leaf : int, default 50
        Minimum observations per leaf.
    n_time_grid_steps : int, default 100
        Size of the default prediction time grid, built from quantiles of
        the observed event times. Becomes :attr:`times_` after fitting.
    ipcw_strategy : {"alternating", "kaplan-meier"}, default "alternating"
        How the censoring distribution $G$ is estimated.
        ``"alternating"`` boosts a second classifier for
        $G(t \mid x)$, retrained against the current incidence model every
        ``n_iter_before_feedback`` iterations — censoring may then depend
        on the covariates. ``"kaplan-meier"`` uses the marginal reverse
        Kaplan-Meier estimate, fitted once up front.
    n_iter_before_feedback : int, default 20
        Boosting iterations between censoring-model refits. Ignored when
        ``ipcw_strategy="kaplan-meier"``.
    seed : int, optional
        Seed for horizon sampling and base-tree randomness.
    n_horizons_per_observation : int, default 3
        Horizons drawn per observation per iteration. Higher values
        reduce gradient variance at proportional cost.

    Attributes
    ----------
    times_ : (k,) array
        Default prediction time grid (event-time quantiles).
    n_causes : int
        Number of competing causes, counted from the causes **present in
        the training data**. A training split missing a rare cause yields
        a model with fewer causes, and :meth:`predict_cif` will reject
        that cause's index. Check it after fitting when working with
        cross-validation folds or heavily imbalanced causes.

    References
    ----------
    Alberge, Maladière, Grisel, Abécassis & Varoquaux (2025). Survival
    models: proper scoring rule and stochastic optimization with
    competing risks. AISTATS.
    """

    _model: HistGradientBoostingClassifier

    def __init__(
        self,
        *,
        hard_zero_fraction: float = 0.1,
        n_iter: int = 100,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        max_depth: int | None = None,
        min_samples_leaf: int = 50,
        n_time_grid_steps: int = 100,
        ipcw_strategy: Literal["alternating", "kaplan-meier"] = "alternating",
        n_iter_before_feedback: int = 20,
        seed: int | None = None,
        n_horizons_per_observation: int = 3,
    ) -> None:
        if ipcw_strategy not in ("alternating", "kaplan-meier"):
            raise ValueError(
                f"invalid ipcw_strategy: {ipcw_strategy!r}; "
                f"expected 'alternating' or 'kaplan-meier'"
            )
        self.hard_zero_fraction = hard_zero_fraction
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_time_grid_steps = n_time_grid_steps
        self.ipcw_strategy = ipcw_strategy
        self.n_iter_before_feedback = n_iter_before_feedback
        self.seed = seed
        self.n_horizons_per_observation = n_horizons_per_observation

    def fit(
        self,
        X: ArrayLike,
        event_time: ArrayLike,
        event_indicator: ArrayLike,
    ) -> SurvivalBoost:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n, d); got shape {X.shape}")
        Y = np.asarray(event_time, dtype=np.float64)
        delta = np.asarray(event_indicator, dtype=np.int64)
        if not (len(X) == len(Y) == len(delta)):
            raise ValueError(
                f"length mismatch: X has {len(X)} rows, event_time {len(Y)}, "
                f"event_indicator {len(delta)}"
            )

        event_ids = np.unique(np.append(delta, 0))
        self.n_causes = int((event_ids != 0).sum())
        any_event = delta > 0
        self.times_ = _default_time_grid(Y[any_event], self.n_time_grid_steps)

        model = _build_classifier(
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            seed=self.seed,
        )
        censoring = _CensoringModel(Y, any_event, seed=self.seed)
        rng = np.random.RandomState(self.seed)
        alternating = self.ipcw_strategy == "alternating"

        n = len(Y)
        t_max = float(Y.max())
        ipcw_duration = censoring.ipcw_at(Y, X)

        for m in range(self.n_iter):
            features, targets, weights = [], [], []
            for _ in range(self.n_horizons_per_observation):
                horizons = _draw_horizons(rng, n, t_max, self.hard_zero_fraction)
                observed_before, weight = _horizon_targets(
                    any_event,
                    Y,
                    horizons,
                    ipcw_duration,
                    censoring.ipcw_at(horizons, X),
                )
                features.append(np.hstack([horizons[:, None], X]))
                targets.append(np.where(observed_before, delta, 0))
                weights.append(weight)

            model.max_iter += 1
            model.fit(
                np.vstack(features),
                np.concatenate(targets),
                sample_weight=np.concatenate(weights),
            )
            if not np.array_equal(model.classes_, event_ids):
                raise ValueError(
                    f"horizon resampling left some causes unobserved at "
                    f"iteration {m}; consider lowering hard_zero_fraction "
                    f"(currently {self.hard_zero_fraction})"
                )

            if alternating and m % self.n_iter_before_feedback == 0:
                censoring.refit(
                    model,
                    X,
                    Y,
                    any_event,
                    rng,
                    hard_zero_fraction=self.hard_zero_fraction,
                    n_rounds=self.n_iter_before_feedback,
                )
                ipcw_duration = censoring.ipcw_at(Y, X)

        self._model = model
        return self

    def _cif(
        self,
        X: ArrayLike,
        times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        # One classifier call over the (time, X) product instead of one per
        # horizon; predictions are row-independent, so the results match.
        stacked = np.hstack([np.repeat(times, n)[:, None], np.tile(X, (len(times), 1))])
        proba = self._model.predict_proba(stacked)
        cif = proba.reshape(len(times), n, -1).transpose(1, 2, 0)[:, 1:, :]
        # Horizon-sampled boosting carries no monotonicity constraint, so
        # the raw estimate can dip on fine grids.
        return np.maximum.accumulate(cif, axis=-1)


class _CensoringModel:
    """Censoring survival $\\hat G$ and the IPCW weights derived from it.

    Starts from the marginal reverse Kaplan-Meier estimate. Under the
    alternating strategy, :meth:`refit` boosts a classifier for
    $G(t \\mid x)$ against the current incidence model (Alberge et al.,
    2025, Algorithm 3) and later weights become covariate-dependent.
    """

    def __init__(
        self,
        Y: NDArray[np.float64],
        any_event: NDArray[np.bool_],
        seed: int | None,
    ) -> None:
        self.G_marginal = censoring_distribution(Y, any_event)
        positive = self.G_marginal.value[self.G_marginal.value > 0]
        self.min_prob = max(float(positive.min()), _MIN_IPCW_PROB)
        self.seed = seed
        self.model: HistGradientBoostingClassifier | None = None

    def ipcw_at(
        self, times: NDArray[np.float64], X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """$1 / \\hat G(t \\mid x)$, clipped away from division blow-up."""
        if self.model is None:
            G = self.G_marginal(times)
        else:
            G = self.model.predict_proba(np.hstack([times[:, None], X]))[:, 0]
        return 1.0 / np.clip(G, self.min_prob, 1.0)

    def refit(
        self,
        incidence_model: HistGradientBoostingClassifier,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        any_event: NDArray[np.bool_],
        rng: np.random.RandomState,
        *,
        hard_zero_fraction: float,
        n_rounds: int,
    ) -> None:
        censored = ~any_event
        inv_surv_duration = self._inverse_survival(incidence_model, Y, X)
        if self.model is None:
            # The censoring booster uses the reference implementation's
            # fixed hyperparameters, not the incidence model's.
            self.model = _build_classifier(
                learning_rate=0.05,
                max_leaf_nodes=31,
                max_depth=None,
                min_samples_leaf=50,
                seed=self.seed,
            )
        t_max = float(Y.max())
        for _ in range(n_rounds):
            horizons = _draw_horizons(rng, len(Y), t_max, hard_zero_fraction)
            observed_before, weight = _horizon_targets(
                censored,
                Y,
                horizons,
                inv_surv_duration,
                self._inverse_survival(incidence_model, horizons, X),
            )
            self.model.max_iter += 1
            self.model.fit(
                np.hstack([horizons[:, None], X]),
                observed_before.astype(np.int64),
                sample_weight=weight,
            )

    def _inverse_survival(
        self,
        incidence_model: HistGradientBoostingClassifier,
        times: NDArray[np.float64],
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        S: NDArray[np.float64] = incidence_model.predict_proba(
            np.hstack([times[:, None], X])
        )[:, 0]
        return 1.0 / np.clip(S, self.min_prob, 1.0)


def _horizon_targets(
    event: NDArray[np.bool_],
    Y: NDArray[np.float64],
    horizons: NDArray[np.float64],
    ipcw_duration: NDArray[np.float64],
    ipcw_horizons: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Binary target and IPCW weight at per-observation horizons.

    The target is 1 when the event occurred by the horizon. Subjects still
    under observation at the horizon contribute 0 with weight
    ``ipcw_horizons``; subjects whose event occurred earlier contribute 1
    with weight ``ipcw_duration``; everyone else (lost before the horizon)
    gets weight 0. Alberge et al. (2025), Algorithm 2.
    """
    observed_before = event & (horizons >= Y)
    weight = np.where(horizons < Y, ipcw_horizons, 0.0)
    weight = np.where(observed_before, ipcw_duration, weight)
    return observed_before, weight


def _draw_horizons(
    rng: np.random.RandomState,
    n: int,
    t_max: float,
    hard_zero_fraction: float,
) -> NDArray[np.float64]:
    horizons = rng.uniform(0.0, t_max, n)
    n_zeros = max(int(hard_zero_fraction * n), 1)
    horizons[rng.choice(n, n_zeros, replace=False)] = 0.0
    return horizons


def _default_time_grid(
    observed_times: NDArray[np.float64], n_steps: int
) -> NDArray[np.float64]:
    if len(observed_times) > n_steps:
        return np.quantile(observed_times, np.linspace(0.0, 1.0, num=n_steps))
    return np.sort(observed_times)


def _build_classifier(
    *,
    learning_rate: float,
    max_leaf_nodes: int,
    max_depth: int | None,
    min_samples_leaf: int,
    seed: int | None,
) -> HistGradientBoostingClassifier:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as exc:
        raise ImportError(
            "SurvivalBoost requires the optional 'scikit-learn' dependency; "
            "install it with `pip install tausurv[boost]`"
        ) from exc
    # max_iter starts at 1 and is incremented before every warm-started
    # fit call, so the first call grows two trees — kept for parity with
    # the reference implementation.
    return HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=1,
        warm_start=True,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )
