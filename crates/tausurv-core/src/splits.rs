//! Split-criterion trait and concrete implementations.

use std::cmp::Ordering;

use ndarray::ArrayView1;

use crate::data::SurvivalData;

/// The best valid split found for a node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BestSplit {
    /// Feature index split on.
    pub feature: u32,
    /// Threshold value: samples with `x[feature] <= threshold` go left.
    pub threshold: f64,
    /// Criterion score at this split (higher = better).
    pub score: f64,
}

/// A rule for choosing the best binary split at a node.
///
/// Implementations own the per-node search (feature iteration, threshold
/// enumeration, scoring) — the tree just receives the chosen [`BestSplit`].
/// This lets each criterion use whatever algorithmic shape it prefers
/// (naive recompute, sorted streaming, gradient pseudo-outcomes).
pub trait SplitCriterion: Sync {
    /// Find the best split among `candidate_features` for the given node.
    ///
    /// Returns `None` if no candidate satisfies the leaf-size constraint or
    /// yields a positive variance term.
    fn find_best_split(
        &self,
        data: SurvivalData<'_>,
        sample_idx: &[u32],
        candidate_features: &[u32],
        min_samples_leaf: usize,
    ) -> Option<BestSplit>;
}

/// Log-rank chi-square splitting (Ishwaran et al., 2008).
///
/// $$
/// Z = \frac{\big(\sum_j (O_{L,j} - E_{L,j})\big)^2}{\sum_j V_{L,j}}
/// $$
///
/// Mirrors `tausurv.trees.survival_tree._log_rank_statistic`: same
/// variance formula, same $N(\tau) > 1$ guard, same strict
/// `score > best_score` tie-breaking. Higher score = better split.
///
/// The criterion is stateless — it reads `y` and `delta` from the
/// [`SurvivalData`] passed at fit time.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogRankCriterion;

impl SplitCriterion for LogRankCriterion {
    fn find_best_split(
        &self,
        data: SurvivalData<'_>,
        sample_idx: &[u32],
        candidate_features: &[u32],
        min_samples_leaf: usize,
    ) -> Option<BestSplit> {
        let mut best: Option<BestSplit> = None;
        let mut best_score = f64::NEG_INFINITY;

        for &feat in candidate_features {
            let col = data.x.column(feat as usize);
            for thresh in unique_midpoints(col, sample_idx) {
                let (n_left, n_right) = count_left_right(col, sample_idx, thresh);
                if n_left < min_samples_leaf || n_right < min_samples_leaf {
                    continue;
                }
                let score = log_rank_score(data.y, data.delta, sample_idx, col, thresh);
                if score > best_score {
                    best_score = score;
                    best = Some(BestSplit { feature: feat, threshold: thresh, score });
                }
            }
        }

        best
    }
}

/// Midpoints of consecutive unique values of `col` over `sample_idx`.
fn unique_midpoints(col: ArrayView1<f64>, sample_idx: &[u32]) -> Vec<f64> {
    let mut vals: Vec<f64> = sample_idx.iter().map(|&i| col[i as usize]).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    vals.dedup_by(|a, b| a == b);
    if vals.len() < 2 {
        return Vec::new();
    }
    vals.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect()
}

fn count_left_right(
    col: ArrayView1<f64>,
    sample_idx: &[u32],
    threshold: f64,
) -> (usize, usize) {
    let n_left = sample_idx
        .iter()
        .filter(|&&i| col[i as usize] <= threshold)
        .count();
    (n_left, sample_idx.len() - n_left)
}

/// Single-pass log-rank chi-square on the subset.
///
/// Allocates one `Vec<(f64, u8, u8)>` of length `sample_idx.len()` to sort
/// by observed time, then walks unique times in order accumulating
/// $\sum (O_L - E_L)$ and $\sum V_L$. O(n log n) per call.
fn log_rank_score(
    y: &[f64],
    delta: &[u8],
    sample_idx: &[u32],
    feature_col: ArrayView1<f64>,
    threshold: f64,
) -> f64 {
    let n = sample_idx.len();
    if n == 0 {
        return 0.0;
    }

    // (Y_i, delta_i, left_i) per sample, sorted by Y.
    let mut rows: Vec<(f64, u8, u8)> = sample_idx
        .iter()
        .map(|&i| {
            let i = i as usize;
            let left = u8::from(feature_col[i] <= threshold);
            (y[i], delta[i], left)
        })
        .collect();
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let mut at_risk_total = n;
    let mut at_risk_left = rows.iter().map(|r| r.2 as usize).sum::<usize>();
    let mut obs_minus_exp = 0.0_f64;
    let mut var_sum = 0.0_f64;
    let mut i = 0;

    while i < n {
        let t = rows[i].0;
        let mut d_total = 0_u32;
        let mut d_left = 0_u32;
        let mut n_at = 0_u32;
        let mut n_at_left = 0_u32;
        while i < n && rows[i].0 == t {
            n_at += 1;
            n_at_left += u32::from(rows[i].2);
            if rows[i].1 == 1 {
                d_total += 1;
                d_left += u32::from(rows[i].2);
            }
            i += 1;
        }

        if d_total > 0 {
            let p = at_risk_left as f64 / at_risk_total as f64;
            let e_left = f64::from(d_total) * p;
            obs_minus_exp += f64::from(d_left) - e_left;
            if at_risk_total > 1 {
                let n_v = at_risk_total as f64;
                let d_v = f64::from(d_total);
                var_sum += d_v * p * (1.0 - p) * (n_v - d_v) / (n_v - 1.0);
            }
        }

        at_risk_total -= n_at as usize;
        at_risk_left -= n_at_left as usize;
    }

    if var_sum <= 0.0 {
        return 0.0;
    }
    obs_minus_exp * obs_minus_exp / var_sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;

    fn data<'a>(
        x: &'a Array2<f64>,
        y: &'a [f64],
        delta: &'a [u8],
    ) -> SurvivalData<'a> {
        SurvivalData::new(x.view(), y, delta)
    }

    /// Two groups perfectly separated by Y: log-rank should be very large.
    #[test]
    fn perfect_separation_high_score() {
        let n = 20;
        let mut y = vec![0.0; n];
        let delta = vec![1_u8; n];
        let mut x = Array2::<f64>::zeros((n, 1));
        for i in 0..n / 2 {
            y[i] = 1.0 + i as f64;
            x[(i, 0)] = 0.0;
        }
        for i in n / 2..n {
            y[i] = 100.0 + i as f64;
            x[(i, 0)] = 1.0;
        }
        let idx: Vec<u32> = (0..n as u32).collect();
        let split = LogRankCriterion
            .find_best_split(data(&x, &y, &delta), &idx, &[0], 2)
            .expect("a split should exist");
        assert_eq!(split.feature, 0);
        assert!(split.threshold > 0.0 && split.threshold < 1.0);
        assert!(split.score > 10.0, "score={}", split.score);
    }

    /// Random feature with no signal: best split's score should be small.
    #[test]
    fn no_signal_low_score() {
        let n = 40;
        let y: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let delta = vec![1_u8; n];
        let x_data: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
        let idx: Vec<u32> = (0..n as u32).collect();
        let split = LogRankCriterion
            .find_best_split(data(&x, &y, &delta), &idx, &[0], 5);
        if let Some(s) = split {
            assert!(s.score < 5.0, "no-signal score unexpectedly high: {}", s.score);
        }
    }

    /// Subset semantics: feeding only half the samples ignores the rest.
    #[test]
    fn respects_sample_idx_subset() {
        let n = 20;
        let y: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let delta = vec![1_u8; n];
        let x_data: Vec<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }).collect();
        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();

        let idx: Vec<u32> = (0..(n / 2) as u32).collect();
        let split = LogRankCriterion
            .find_best_split(data(&x, &y, &delta), &idx, &[0], 1);
        assert!(split.is_none(), "single-valued feature should give no split");
    }

    #[test]
    fn constant_feature_no_split() {
        let n = 10;
        let y: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let delta = vec![1_u8; n];
        let x = Array2::<f64>::zeros((n, 1));
        let idx: Vec<u32> = (0..n as u32).collect();
        assert!(
            LogRankCriterion
                .find_best_split(data(&x, &y, &delta), &idx, &[0], 1)
                .is_none()
        );
    }

    #[test]
    fn all_censored_zero_score() {
        let n = 10;
        let y: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let delta = vec![0_u8; n];
        let x_data: Vec<f64> = (0..n).map(|i| (i / 5) as f64).collect();
        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
        let idx: Vec<u32> = (0..n as u32).collect();
        let split = LogRankCriterion
            .find_best_split(data(&x, &y, &delta), &idx, &[0], 1);
        if let Some(s) = split {
            assert_relative_eq!(s.score, 0.0);
        }
    }
}
