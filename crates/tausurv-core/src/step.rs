//! Right- or left-continuous step function on a sorted time grid.
//!
//! Mirror of the Python `tausurv.step.StepFunction`.

use std::cmp::Ordering;

/// Default continuity used when evaluating the step function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Right-continuous (jumps at grid points are taken at the grid value).
    Right,
    /// Left-continuous (jumps are taken at the prior grid value).
    Left,
}

/// Step function on a strictly increasing time grid.
///
/// Below `time[0]` the function returns `baseline`; at and above `time[k]` it
/// returns `value[k]`, with the boundary behaviour controlled by [`Side`].
#[derive(Debug, Clone)]
pub struct StepFunction {
    time: Vec<f64>,
    value: Vec<f64>,
    side: Side,
    baseline: f64,
}

impl StepFunction {
    /// Construct from sorted grid points and values.
    ///
    /// # Panics
    /// In debug builds, if `time` is not strictly increasing or lengths differ.
    pub fn new(time: Vec<f64>, value: Vec<f64>, side: Side, baseline: f64) -> Self {
        debug_assert_eq!(time.len(), value.len(), "time and value must align");
        debug_assert!(
            time.windows(2).all(|w| w[0] < w[1]),
            "time must be strictly increasing",
        );
        Self {
            time,
            value,
            side,
            baseline,
        }
    }

    /// Empty step function: always returns `baseline`.
    pub fn empty(side: Side, baseline: f64) -> Self {
        Self {
            time: Vec::new(),
            value: Vec::new(),
            side,
            baseline,
        }
    }

    /// Grid times.
    pub fn time(&self) -> &[f64] {
        &self.time
    }

    /// Grid values.
    pub fn value(&self) -> &[f64] {
        &self.value
    }

    /// Default continuity.
    pub fn side(&self) -> Side {
        self.side
    }

    /// Baseline value returned below `time[0]`.
    pub fn baseline(&self) -> f64 {
        self.baseline
    }

    /// Evaluate at a single point using the default `side`.
    pub fn at(&self, q: f64) -> f64 {
        self.at_with(q, self.side)
    }

    /// Evaluate at a single point with explicit continuity.
    ///
    /// Mirrors numpy `searchsorted(side=...)`-then-minus-one:
    /// `side=Right` returns the value of the latest grid point ≤ `q`,
    /// `side=Left` returns the value of the latest grid point < `q`.
    pub fn at_with(&self, q: f64, side: Side) -> f64 {
        if self.time.is_empty() {
            return self.baseline;
        }
        let idx = match side {
            Side::Right => searchsorted_right(&self.time, q),
            Side::Left => searchsorted_left(&self.time, q),
        };
        if idx == 0 {
            self.baseline
        } else {
            self.value[idx - 1]
        }
    }

    /// Evaluate at every point in `queries`, writing into `out`.
    ///
    /// `out` must have the same length as `queries`. Default continuity.
    pub fn eval_into(&self, queries: &[f64], out: &mut [f64]) {
        debug_assert_eq!(queries.len(), out.len());
        for (q, slot) in queries.iter().zip(out.iter_mut()) {
            *slot = self.at(*q);
        }
    }
}

/// `np.searchsorted(arr, q, side="right")` — first index `i` with `arr[i] > q`.
fn searchsorted_right(arr: &[f64], q: f64) -> usize {
    arr.partition_point(|&x| !matches!(x.partial_cmp(&q), Some(Ordering::Greater)))
}

/// `np.searchsorted(arr, q, side="left")` — first index `i` with `arr[i] >= q`.
fn searchsorted_left(arr: &[f64], q: f64) -> usize {
    arr.partition_point(|&x| matches!(x.partial_cmp(&q), Some(Ordering::Less)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn fixture() -> StepFunction {
        StepFunction::new(vec![1.0, 2.0, 3.0], vec![0.1, 0.4, 0.9], Side::Right, 0.0)
    }

    #[test]
    fn below_grid_returns_baseline() {
        let f = fixture();
        assert_eq!(f.at(0.5), 0.0);
        assert_eq!(f.at(-1e9), 0.0);
    }

    #[test]
    fn right_continuous_at_grid_point() {
        let f = fixture();
        assert_relative_eq!(f.at(1.0), 0.1);
        assert_relative_eq!(f.at(2.0), 0.4);
        assert_relative_eq!(f.at(3.0), 0.9);
    }

    #[test]
    fn left_continuous_at_grid_point() {
        let f = fixture();
        assert_eq!(f.at_with(1.0, Side::Left), 0.0); // below first jump
        assert_relative_eq!(f.at_with(2.0, Side::Left), 0.1);
        assert_relative_eq!(f.at_with(3.0, Side::Left), 0.4);
    }

    #[test]
    fn between_grid_points_holds_latest() {
        let f = fixture();
        assert_relative_eq!(f.at(1.5), 0.1);
        assert_relative_eq!(f.at(2.99), 0.4);
        assert_relative_eq!(f.at(100.0), 0.9);
    }

    #[test]
    fn empty_returns_baseline_everywhere() {
        let f = StepFunction::empty(Side::Right, 0.0);
        assert_eq!(f.at(-1.0), 0.0);
        assert_eq!(f.at(0.0), 0.0);
        assert_eq!(f.at(1e9), 0.0);
    }

    #[test]
    fn eval_into_batch_matches_pointwise() {
        let f = fixture();
        let queries = [0.5, 1.0, 1.5, 2.5, 100.0];
        let mut out = [0.0; 5];
        f.eval_into(&queries, &mut out);
        for (i, q) in queries.iter().enumerate() {
            assert_eq!(out[i], f.at(*q));
        }
    }
}
