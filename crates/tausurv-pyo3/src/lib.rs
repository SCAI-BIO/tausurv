//! PyO3 bindings for `tausurv-core`.
//!
//! Compiled to the `tausurv._tausurv_core` Python extension module. All
//! `#[pymodule]` / `#[pyfunction]` / `#[pyclass]` code lives here; the
//! core crate stays Python-agnostic.

use ndarray::{ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tausurv_core::{LogRankCriterion, MaxFeatures, SurvivalData, Tree, TreeConfig};

#[pyfunction]
fn version() -> &'static str {
    tausurv_core::VERSION
}

/// Immutable handle to a fitted log-rank survival tree.
#[pyclass(name = "LogRankSurvivalTree", frozen, module = "tausurv._tausurv_core")]
pub struct PyLogRankSurvivalTree {
    inner: Tree,
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
#[pyfunction]
#[pyo3(signature = (
    X, event_time, event_indicator, *,
    min_samples_leaf = 15,
    max_depth = None,
    max_features = None,
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

    let data = SurvivalData::new(x_view, y, delta);
    let config = TreeConfig {
        min_samples_leaf,
        max_depth,
        max_features: parse_max_features(max_features.as_ref(), p as u32)?,
    };
    let seed = seed.unwrap_or(0);

    let tree =
        py.detach(|| Tree::fit(data, &LogRankCriterion, &config, seed));
    Ok(PyLogRankSurvivalTree { inner: tree })
}

fn parse_max_features(
    value: Option<&Bound<'_, PyAny>>,
    n_features: u32,
) -> PyResult<MaxFeatures> {
    let Some(v) = value else {
        return Ok(MaxFeatures::All);
    };
    if v.is_none() {
        return Ok(MaxFeatures::All);
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

#[pymodule]
fn _tausurv_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(fit_log_rank_tree, m)?)?;
    m.add_class::<PyLogRankSurvivalTree>()?;
    Ok(())
}
