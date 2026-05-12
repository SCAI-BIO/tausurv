//! Survival-analysis input data: covariates, observed times, event indicators.

use ndarray::ArrayView2;

/// Borrowed view over a survival dataset.
///
/// The "input trio" — covariates `x`, observed times `y`, event indicators
/// `delta` — appears together in every survival routine, so we bundle it
/// once. `delta[i] == 1` means the event was observed; `0` means censored.
///
/// All three components must align: `x.shape()[0] == y.len() == delta.len()`.
/// This is checked in [`SurvivalData::new`].
#[derive(Debug, Clone, Copy)]
pub struct SurvivalData<'a> {
    /// Covariate matrix, shape `(n_samples, n_features)`. Layout is not
    /// constrained at this level; callers that care (e.g. split-criterion
    /// inner loops) should arrange F-contiguous storage at the boundary.
    pub x: ArrayView2<'a, f64>,
    /// Observed times $Y_i = \min(T_i, C_i)$.
    pub y: &'a [f64],
    /// Event indicators ($1$ = event observed, $0$ = censored).
    pub delta: &'a [u8],
}

impl<'a> SurvivalData<'a> {
    /// Bundle the three components, checking that lengths align.
    pub fn new(x: ArrayView2<'a, f64>, y: &'a [f64], delta: &'a [u8]) -> Self {
        assert_eq!(
            x.shape()[0],
            y.len(),
            "x has {} rows but y has {} elements",
            x.shape()[0],
            y.len(),
        );
        assert_eq!(
            y.len(),
            delta.len(),
            "y has {} elements but delta has {}",
            y.len(),
            delta.len(),
        );
        Self { x, y, delta }
    }

    /// Number of samples (rows of `x`).
    pub fn n_samples(&self) -> usize {
        self.x.shape()[0]
    }

    /// Number of covariates (columns of `x`).
    pub fn n_features(&self) -> usize {
        self.x.shape()[1]
    }
}
