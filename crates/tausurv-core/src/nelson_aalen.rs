//! Nelson-Aalen cumulative-hazard estimator on a sample subset.
//!
//! $$ \hat\Lambda(t) = \sum_{t_i \le t,\, \delta_i = 1} \frac{d_i}{n_i} $$
//!
//! with $d_i$ events at the $i$-th unique observed time and $n_i$ the
//! at-risk count just before it. The returned step function only jumps at
//! event times (censored-only ties are absorbed into the at-risk count
//! without adding a grid point — the function evaluates identically to the
//! all-unique-time representation but stores fewer breakpoints).

use std::cmp::Ordering;

use crate::step::{Side, StepFunction};

/// Nelson-Aalen on the samples in `sample_idx`.
///
/// `y` and `delta` are the full training arrays; `sample_idx` selects the
/// rows that contribute to this estimate. Pass `(0..n)` for the whole
/// cohort; pass a leaf's sample set when computing leaf hazards.
///
/// # Panics
/// In debug builds, if `y.len() != delta.len()` or any sample index is out
/// of range.
pub fn nelson_aalen(y: &[f64], delta: &[u8], sample_idx: &[u32]) -> StepFunction {
    debug_assert_eq!(y.len(), delta.len(), "y and delta must align");

    if sample_idx.is_empty() {
        return StepFunction::empty(Side::Right, 0.0);
    }

    let mut pairs: Vec<(f64, u8)> = sample_idx
        .iter()
        .map(|&i| {
            let i = i as usize;
            (y[i], delta[i])
        })
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let n = pairs.len();
    let mut times = Vec::new();
    let mut hazard = Vec::new();
    let mut cum = 0.0_f64;
    let mut at_risk = n;
    let mut i = 0;

    while i < n {
        let t = pairs[i].0;
        let mut d_at = 0_u32;
        let mut n_at = 0_u32;
        while i < n && pairs[i].0 == t {
            n_at += 1;
            d_at += u32::from(pairs[i].1 == 1);
            i += 1;
        }
        if d_at > 0 && at_risk > 0 {
            cum += f64::from(d_at) / at_risk as f64;
            times.push(t);
            hazard.push(cum);
        }
        at_risk -= n_at as usize;
    }

    StepFunction::new(times, hazard, Side::Right, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Hand-computed example.
    ///
    /// Times: [1, 2, 2, 3, 4, 5], events: [1, 1, 0, 1, 0, 1].
    /// At t=1: d=1, n=6 -> +1/6
    /// At t=2: d=1, n=5 (1 event + 1 cens drop out)
    /// At t=3: d=1, n=3
    /// At t=5: d=1, n=1
    /// (t=4 is censored only, no jump but reduces at-risk.)
    #[test]
    fn matches_hand_computation() {
        let y = vec![1.0, 2.0, 2.0, 3.0, 4.0, 5.0];
        let delta = vec![1, 1, 0, 1, 0, 1];
        let idx: Vec<u32> = (0..6).collect();
        let f = nelson_aalen(&y, &delta, &idx);

        let expected_times = [1.0, 2.0, 3.0, 5.0];
        let expected_jumps = [1.0 / 6.0, 1.0 / 5.0, 1.0 / 3.0, 1.0 / 1.0];
        let mut expected_cum = 0.0;
        for (i, &t) in expected_times.iter().enumerate() {
            expected_cum += expected_jumps[i];
            assert_relative_eq!(f.at(t), expected_cum, max_relative = 1e-12);
        }

        assert_eq!(f.at(0.5), 0.0); // baseline below first event
        assert_relative_eq!(f.at(4.0), expected_jumps[..3].iter().sum::<f64>());
    }

    #[test]
    fn all_censored_gives_flat_zero() {
        let y = vec![1.0, 2.0, 3.0];
        let delta = vec![0, 0, 0];
        let idx = vec![0, 1, 2];
        let f = nelson_aalen(&y, &delta, &idx);
        assert!(f.time().is_empty());
        assert_eq!(f.at(2.5), 0.0);
        assert_eq!(f.at(100.0), 0.0);
    }

    #[test]
    fn single_event_at_time_zero_safe() {
        let y = vec![0.0];
        let delta = vec![1];
        let idx = vec![0];
        let f = nelson_aalen(&y, &delta, &idx);
        assert_relative_eq!(f.at(0.0), 1.0);
        assert_relative_eq!(f.at(0.5), 1.0);
    }

    #[test]
    fn subset_uses_only_selected_indices() {
        let y = vec![1.0, 99.0, 2.0, 99.0, 3.0];
        let delta = vec![1, 1, 1, 1, 1];
        let idx = vec![0, 2, 4]; // skip the 99s
        let f = nelson_aalen(&y, &delta, &idx);
        assert_relative_eq!(f.at(1.0), 1.0 / 3.0);
        assert_relative_eq!(f.at(2.0), 1.0 / 3.0 + 1.0 / 2.0);
        assert_relative_eq!(f.at(3.0), 1.0 / 3.0 + 1.0 / 2.0 + 1.0);
    }
}
