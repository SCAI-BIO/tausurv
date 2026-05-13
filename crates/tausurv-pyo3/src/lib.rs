//! PyO3 bindings for `tausurv-core`.
//!
//! Compiled to the `tausurv._tausurv_core` Python extension module. All
//! `#[pymodule]` / `#[pyfunction]` / `#[pyclass]` code lives here; the
//! core crate stays Python-agnostic.

use ndarray::{ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tausurv_core::{
    Bootstrap, Forest, ForestConfig, GradientCriterion, HonestyMode, LogRankCriterion,
    MaxFeatures, StepFunction, SurvivalData, Tree, TreeConfig,
};

#[pyfunction]
fn version() -> &'static str {
    tausurv_core::VERSION
}

/// Immutable handle to a fitted log-rank survival tree.
#[pyclass(name = "LogRankSurvivalTree", frozen, module = "tausurv._tausurv_core")]
pub struct PyLogRankSurvivalTree {
    inner: Tree<StepFunction>,
}

#[pymethods]
impl PyLogRankSurvivalTree {
    #[getter]
    fn n_features(&self) -> usize {
        self.inner.n_features()
    }

    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.n_nodes()
    }

    #[getter]
    fn n_leaves(&self) -> usize {
        self.inner.n_leaves()
    }

    #[getter]
    fn depth(&self) -> u32 {
        self.inner.depth()
    }

    fn __repr__(&self) -> String {
        format!(
            "LogRankSurvivalTree(n_features={}, n_leaves={}, depth={})",
            self.n_features(),
            self.n_leaves(),
            self.depth(),
        )
    }

    /// Predict the cumulative hazard $\hat\Lambda(t \mid x)$ for each row
    /// of `X` at every time in `times`. Returns an `(n, k)` array.
    #[pyo3(signature = (X, times))]
    #[allow(non_snake_case)] // `X` is the public Python kwarg name.
    fn predict_cumulative_hazard<'py>(
        &self,
        py: Python<'py>,
        X: PyReadonlyArray2<'py, f64>,
        times: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_view = X.as_array();
        if x_view.shape()[1] != self.inner.n_features() {
            return Err(PyValueError::new_err(format!(
                "X has {} features but tree was fit on {}",
                x_view.shape()[1],
                self.inner.n_features(),
            )));
        }
        let times_slice = times.as_slice()?;

        let out =
            py.detach(|| self.inner.predict_cumulative_hazard(x_view, times_slice));
        Ok(out.into_pyarray(py))
    }
}

/// Fit a single log-rank survival tree.
///
/// When `honesty=True`, the training samples are shuffled and split into a
/// splitting set (drives the log-rank score at each node) and an estimation
/// set (drives the Nelson-Aalen leaf hazard). See Wager & Athey (2018) and
/// Athey, Tibshirani & Wager (2019).
#[pyfunction]
#[pyo3(signature = (
    X, event_time, event_indicator, *,
    min_samples_leaf = 15,
    max_depth = None,
    max_features = None,
    honesty = false,
    honesty_fraction = 0.5,
    seed = None,
))]
#[allow(non_snake_case)] // `X` is the public Python kwarg name.
fn fit_log_rank_tree<'py>(
    py: Python<'py>,
    X: PyReadonlyArray2<'py, f64>,
    event_time: PyReadonlyArray1<'py, f64>,
    event_indicator: PyReadonlyArray1<'py, u8>,
    min_samples_leaf: usize,
    max_depth: Option<u32>,
    max_features: Option<Bound<'py, PyAny>>,
    honesty: bool,
    honesty_fraction: f64,
    seed: Option<u64>,
) -> PyResult<PyLogRankSurvivalTree> {
    let x_view = X.as_array();
    let n = x_view.shape()[0];
    let p = x_view.shape()[1];

    let y = event_time.as_slice()?;
    let delta = event_indicator.as_slice()?;
    if y.len() != n || delta.len() != n {
        return Err(PyValueError::new_err(format!(
            "X has {n} rows but event_time has {} and event_indicator has {}",
            y.len(),
            delta.len(),
        )));
    }
    if !is_column_major(&x_view) {
        return Err(PyValueError::new_err(
            "X must be Fortran-contiguous (column-major) f64. \
             Call np.asfortranarray(x, dtype=np.float64) before fitting.",
        ));
    }

    let honesty = if honesty {
        if !(honesty_fraction > 0.0 && honesty_fraction < 1.0) {
            return Err(PyValueError::new_err(format!(
                "honesty_fraction must lie in (0, 1), got {honesty_fraction}"
            )));
        }
        HonestyMode::Split { fraction: honesty_fraction }
    } else {
        HonestyMode::None
    };

    let data = SurvivalData::new(x_view, y, delta);
    let config = TreeConfig {
        min_samples_leaf,
        max_depth,
        max_features: parse_max_features(
            max_features.as_ref(), p as u32, MaxFeatures::All,
        )?,
        honesty,
    };
    let seed = seed.unwrap_or(0);

    let tree =
        py.detach(|| Tree::fit(data, &LogRankCriterion, &config, seed));
    Ok(PyLogRankSurvivalTree { inner: tree })
}

fn parse_max_features(
    value: Option<&Bound<'_, PyAny>>,
    n_features: u32,
    default: MaxFeatures,
) -> PyResult<MaxFeatures> {
    let Some(v) = value else {
        return Ok(default);
    };
    if v.is_none() {
        return Ok(default);
    }
    if let Ok(s) = v.extract::<String>() {
        return match s.as_str() {
            "sqrt" => Ok(MaxFeatures::Sqrt),
            "all" => Ok(MaxFeatures::All),
            other => Err(PyValueError::new_err(format!(
                "max_features={other:?} not recognised; expected None, 'sqrt', 'all', or a positive int"
            ))),
        };
    }
    if let Ok(k) = v.extract::<u32>() {
        if k == 0 {
            return Err(PyValueError::new_err("max_features must be >= 1"));
        }
        if k > n_features {
            return Err(PyValueError::new_err(format!(
                "max_features={k} exceeds n_features={n_features}"
            )));
        }
        return Ok(MaxFeatures::Count(k));
    }
    Err(PyValueError::new_err(
        "max_features must be None, 'sqrt', 'all', or a positive int",
    ))
}

/// Fortran-contiguous check. Single-row or single-column arrays are
/// trivially contiguous in both orders. `ndarray` reports strides in
/// elements, not bytes.
fn is_column_major(view: &ArrayView2<f64>) -> bool {
    if view.len_of(Axis(0)) <= 1 || view.len_of(Axis(1)) <= 1 {
        return true;
    }
    let strides = view.strides();
    strides[0] == 1 && strides[1] == view.shape()[0] as isize
}

/// Immutable handle to a fitted random survival forest.
#[pyclass(name = "LogRankSurvivalForest", frozen, module = "tausurv._tausurv_core")]
pub struct PyLogRankSurvivalForest {
    inner: Forest<StepFunction>,
}

#[pymethods]
impl PyLogRankSurvivalForest {
    #[getter]
    fn n_trees(&self) -> usize {
        self.inner.n_trees()
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.inner.n_features()
    }

    #[getter]
    fn n_train_samples(&self) -> usize {
        self.inner.n_train_samples()
    }

    fn __repr__(&self) -> String {
        format!(
            "LogRankSurvivalForest(n_trees={}, n_features={}, n_train_samples={})",
            self.n_trees(),
            self.n_features(),
            self.n_train_samples(),
        )
    }

    /// Tree-averaged cumulative hazard $\hat\Lambda(t | x)$ for each row of
    /// `X` at every time in `times`. Returns an `(n, k)` array.
    #[pyo3(signature = (X, times))]
    #[allow(non_snake_case)]
    fn predict_cumulative_hazard<'py>(
        &self,
        py: Python<'py>,
        X: PyReadonlyArray2<'py, f64>,
        times: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_view = X.as_array();
        if x_view.shape()[1] != self.inner.n_features() {
            return Err(PyValueError::new_err(format!(
                "X has {} features but forest was fit on {}",
                x_view.shape()[1],
                self.inner.n_features(),
            )));
        }
        let times_slice = times.as_slice()?;
        let out =
            py.detach(|| self.inner.predict_cumulative_hazard(x_view, times_slice));
        Ok(out.into_pyarray(py))
    }

    /// GRF-style forest weights: for each row of `X`, return an
    /// `(n_train,)` weight vector over the training samples.
    ///
    /// $$
    /// w_i(x) = \frac{1}{B} \sum_b \frac{\mathbb{1}\{\mathrm{leaf}_b(x) = \mathrm{leaf}_b(x_i)\}}{|\mathrm{leaf}_b(x)|}
    /// $$
    #[pyo3(signature = (X))]
    #[allow(non_snake_case)]
    fn forest_weights<'py>(
        &self,
        py: Python<'py>,
        X: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_view = X.as_array();
        if x_view.shape()[1] != self.inner.n_features() {
            return Err(PyValueError::new_err(format!(
                "X has {} features but forest was fit on {}",
                x_view.shape()[1],
                self.inner.n_features(),
            )));
        }
        let out = py.detach(|| self.inner.forest_weights_batch(x_view));
        Ok(out.into_pyarray(py))
    }

    /// Forest weights for a single query point. Returns a `(n_train,)`
    /// array — convenience over [`forest_weights`] when there's only one
    /// query.
    #[allow(non_snake_case)]
    fn forest_weights_one<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_view = x.as_array();
        if x_view.len() != self.inner.n_features() {
            return Err(PyValueError::new_err(format!(
                "x has {} features but forest was fit on {}",
                x_view.len(),
                self.inner.n_features(),
            )));
        }
        let out = py.detach(|| self.inner.forest_weights(x_view));
        Ok(out.into_pyarray(py))
    }
}

/// Fit a random survival forest with log-rank splitting.
///
/// Trees are fitted in parallel via `rayon` over per-tree seeds derived
/// deterministically from `seed`. The result is identical across thread
/// counts.
#[pyfunction]
#[pyo3(signature = (
    X, event_time, event_indicator, *,
    n_trees = 100,
    min_samples_leaf = 15,
    max_depth = None,
    max_features = None,
    bootstrap = true,
    subsample_fraction = 1.0,
    honesty = false,
    honesty_fraction = 0.5,
    seed = None,
))]
#[allow(non_snake_case)]
fn fit_log_rank_forest<'py>(
    py: Python<'py>,
    X: PyReadonlyArray2<'py, f64>,
    event_time: PyReadonlyArray1<'py, f64>,
    event_indicator: PyReadonlyArray1<'py, u8>,
    n_trees: usize,
    min_samples_leaf: usize,
    max_depth: Option<u32>,
    max_features: Option<Bound<'py, PyAny>>,
    bootstrap: bool,
    subsample_fraction: f64,
    honesty: bool,
    honesty_fraction: f64,
    seed: Option<u64>,
) -> PyResult<PyLogRankSurvivalForest> {
    let x_view = X.as_array();
    let n = x_view.shape()[0];
    let p = x_view.shape()[1];

    let y = event_time.as_slice()?;
    let delta = event_indicator.as_slice()?;
    if y.len() != n || delta.len() != n {
        return Err(PyValueError::new_err(format!(
            "X has {n} rows but event_time has {} and event_indicator has {}",
            y.len(),
            delta.len(),
        )));
    }
    if !is_column_major(&x_view) {
        return Err(PyValueError::new_err(
            "X must be Fortran-contiguous (column-major) f64. \
             Call np.asfortranarray(x, dtype=np.float64) before fitting.",
        ));
    }
    if !(subsample_fraction > 0.0 && subsample_fraction <= 1.0) {
        return Err(PyValueError::new_err(format!(
            "subsample_fraction must lie in (0, 1], got {subsample_fraction}"
        )));
    }
    if n_trees == 0 {
        return Err(PyValueError::new_err("n_trees must be >= 1"));
    }

    let honesty = if honesty {
        if !(honesty_fraction > 0.0 && honesty_fraction < 1.0) {
            return Err(PyValueError::new_err(format!(
                "honesty_fraction must lie in (0, 1), got {honesty_fraction}"
            )));
        }
        HonestyMode::Split { fraction: honesty_fraction }
    } else {
        HonestyMode::None
    };

    let data = SurvivalData::new(x_view, y, delta);
    let config = ForestConfig {
        n_trees,
        tree: TreeConfig {
            min_samples_leaf,
            max_depth,
            max_features: parse_max_features(
                max_features.as_ref(), p as u32, MaxFeatures::Sqrt,
            )?,
            honesty,
        },
        bootstrap: Bootstrap {
            with_replacement: bootstrap,
            fraction: subsample_fraction,
        },
    };
    let seed = seed.unwrap_or(0);

    let forest =
        py.detach(|| Forest::fit(data, &LogRankCriterion, &config, seed));
    Ok(PyLogRankSurvivalForest { inner: forest })
}

/// Immutable handle to a fitted gradient (partition-only) forest.
///
/// Built on `Forest<()>` — the GRF-style second-stage forest of the causal
/// survival learner. Predictions are not computed from per-leaf payloads
/// (there aren't any) but through [`forest_weights`], which downstream
/// causurv code combines with the per-training pseudo-outcomes.
#[pyclass(name = "GradientForest", frozen, module = "tausurv._tausurv_core")]
pub struct PyGradientForest {
    inner: Forest<()>,
}

#[pymethods]
impl PyGradientForest {
    #[getter]
    fn n_trees(&self) -> usize {
        self.inner.n_trees()
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.inner.n_features()
    }

    #[getter]
    fn n_train_samples(&self) -> usize {
        self.inner.n_train_samples()
    }

    fn __repr__(&self) -> String {
        format!(
            "GradientForest(n_trees={}, n_features={}, n_train_samples={})",
            self.n_trees(),
            self.n_features(),
            self.n_train_samples(),
        )
    }

    /// Batched forest weights. Returns `(n_query, n_train)`.
    #[pyo3(signature = (X))]
    #[allow(non_snake_case)]
    fn forest_weights<'py>(
        &self,
        py: Python<'py>,
        X: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_view = X.as_array();
        if x_view.shape()[1] != self.inner.n_features() {
            return Err(PyValueError::new_err(format!(
                "X has {} features but forest was fit on {}",
                x_view.shape()[1],
                self.inner.n_features(),
            )));
        }
        let out = py.detach(|| self.inner.forest_weights_batch(x_view));
        Ok(out.into_pyarray(py))
    }

    /// Forest weights for a single query point. Returns `(n_train,)`.
    #[allow(non_snake_case)]
    fn forest_weights_one<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_view = x.as_array();
        if x_view.len() != self.inner.n_features() {
            return Err(PyValueError::new_err(format!(
                "x has {} features but forest was fit on {}",
                x_view.len(),
                self.inner.n_features(),
            )));
        }
        let out = py.detach(|| self.inner.forest_weights(x_view));
        Ok(out.into_pyarray(py))
    }
}

/// Fit a partition-only ("gradient") forest on `(X, pseudo_outcome)`.
///
/// Each tree uses variance-reduction splitting on the per-sample
/// pseudo-outcome (no log-rank, no survival times). Predictions come
/// through [`PyGradientForest.forest_weights`], not per-leaf payloads.
///
/// `pseudo_outcome` length must equal `X.shape[0]`.
#[pyfunction]
#[pyo3(signature = (
    X, pseudo_outcome, *,
    n_trees = 500,
    min_samples_leaf = 5,
    max_depth = None,
    max_features = None,
    bootstrap = false,
    subsample_fraction = 0.5,
    honesty = true,
    honesty_fraction = 0.5,
    seed = None,
))]
#[allow(non_snake_case)]
fn fit_gradient_forest<'py>(
    py: Python<'py>,
    X: PyReadonlyArray2<'py, f64>,
    pseudo_outcome: PyReadonlyArray1<'py, f64>,
    n_trees: usize,
    min_samples_leaf: usize,
    max_depth: Option<u32>,
    max_features: Option<Bound<'py, PyAny>>,
    bootstrap: bool,
    subsample_fraction: f64,
    honesty: bool,
    honesty_fraction: f64,
    seed: Option<u64>,
) -> PyResult<PyGradientForest> {
    let x_view = X.as_array();
    let n = x_view.shape()[0];
    let p = x_view.shape()[1];
    let pseudo = pseudo_outcome.as_slice()?;
    if pseudo.len() != n {
        return Err(PyValueError::new_err(format!(
            "X has {n} rows but pseudo_outcome has {}",
            pseudo.len(),
        )));
    }
    if !is_column_major(&x_view) {
        return Err(PyValueError::new_err(
            "X must be Fortran-contiguous (column-major) f64. \
             Call np.asfortranarray(x, dtype=np.float64) before fitting.",
        ));
    }
    if !(subsample_fraction > 0.0 && subsample_fraction <= 1.0) {
        return Err(PyValueError::new_err(format!(
            "subsample_fraction must lie in (0, 1], got {subsample_fraction}"
        )));
    }
    if n_trees == 0 {
        return Err(PyValueError::new_err("n_trees must be >= 1"));
    }

    let honesty = if honesty {
        if !(honesty_fraction > 0.0 && honesty_fraction < 1.0) {
            return Err(PyValueError::new_err(format!(
                "honesty_fraction must lie in (0, 1), got {honesty_fraction}"
            )));
        }
        HonestyMode::Split { fraction: honesty_fraction }
    } else {
        HonestyMode::None
    };

    // The gradient criterion ignores `y` and `delta`; pass placeholder
    // zero-filled arrays so the SurvivalData contract is satisfied.
    let y_placeholder = vec![0.0_f64; n];
    let delta_placeholder = vec![0_u8; n];
    let data = SurvivalData::new(x_view, &y_placeholder, &delta_placeholder);
    let criterion = GradientCriterion::new(pseudo);
    let config = ForestConfig {
        n_trees,
        tree: TreeConfig {
            min_samples_leaf,
            max_depth,
            max_features: parse_max_features(
                max_features.as_ref(), p as u32, MaxFeatures::Sqrt,
            )?,
            honesty,
        },
        bootstrap: Bootstrap {
            with_replacement: bootstrap,
            fraction: subsample_fraction,
        },
    };
    let seed = seed.unwrap_or(0);
    let forest = py.detach(|| Forest::<()>::fit(data, &criterion, &config, seed));
    Ok(PyGradientForest { inner: forest })
}

#[pymodule]
fn _tausurv_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(fit_log_rank_tree, m)?)?;
    m.add_function(wrap_pyfunction!(fit_log_rank_forest, m)?)?;
    m.add_function(wrap_pyfunction!(fit_gradient_forest, m)?)?;
    m.add_class::<PyLogRankSurvivalTree>()?;
    m.add_class::<PyLogRankSurvivalForest>()?;
    m.add_class::<PyGradientForest>()?;
    Ok(())
}
