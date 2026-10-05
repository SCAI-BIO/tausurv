//! Split-criterion trait and concrete implementations.

use std::cmp::Ordering;

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
/// Same variance formula and $N(\tau) > 1$ guard as
/// `tausurv.trees.survival_tree._log_rank_statistic`; same strict
/// `score > best_score` tie-breaking. Higher score = better split.
///
/// Implementation notes:
///   * Sort by `Y` once per node (cached in [`SortedNode`]).
///   * Per feature: sort the node's samples by the feature, walk samples
///     in groups of equal value, flipping the left-mask incrementally —
///     `O(group_size)` per group, `O(n)` total across all thresholds.
///   * Threshold sequence is the midpoints between consecutive unique
///     feature values, in increasing order — matches
///     `tausurv.trees.survival_tree._find_best_split` exactly so bit-exact
///     parity holds.
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
        let n = sample_idx.len();
        if n == 0 {
            return None;
        }
        // No events in this node → log-rank carries no information.
        // Match the pure-NumPy reference's behaviour: refuse to split.
        if !sample_idx.iter().any(|&i| data.delta[i as usize] == 1) {
            return None;
        }

        let sorted = SortedNode::from_subset(data.y, data.delta, sample_idx);
        // y_inv[original_idx] = position in sorted.indices. Sized to the
        // full training set since `sample_idx` carries original-index
        // values into [0, n_total).
        let mut y_inv: Vec<u32> = vec![u32::MAX; data.y.len()];
        for (k, &orig) in sorted.indices.iter().enumerate() {
            y_inv[orig as usize] = k as u32;
        }

        // Per-node scratch reused across features.
        let mut left_mask: Vec<u8> = vec![0; n];
        let mut feat_sorted: Vec<u32> = Vec::with_capacity(n);

        let mut best: Option<BestSplit> = None;
        let mut best_score = f64::NEG_INFINITY;

        for &feat in candidate_features {
            let col = data.x.column(feat as usize);

            // Sort node samples by feature value.
            feat_sorted.clear();
            feat_sorted.extend_from_slice(sample_idx);
            feat_sorted.sort_by(|&a, &b| {
                col[a as usize]
                    .partial_cmp(&col[b as usize])
                    .unwrap_or(Ordering::Equal)
            });

            // Reset left-mask state for this feature.
            left_mask.fill(0);
            let mut n_left = 0_usize;

            // Walk the sorted samples in groups of equal feature value.
            // After absorbing a group into the left side, evaluate the
            // midpoint between this group's value and the next.
            let mut i = 0;
            while i < n {
                let group_val = col[feat_sorted[i] as usize];
                while i < n && col[feat_sorted[i] as usize] == group_val {
                    let yp = y_inv[feat_sorted[i] as usize] as usize;
                    left_mask[yp] = 1;
                    n_left += 1;
                    i += 1;
                }
                if i == n {
                    break; // last group — no threshold after it
                }
                let next_val = col[feat_sorted[i] as usize];
                let threshold = 0.5 * (group_val + next_val);
                let n_right = n - n_left;
                if n_left < min_samples_leaf || n_right < min_samples_leaf {
                    continue;
                }
                let score = sorted.log_rank_score(&left_mask);
                if score > best_score {
                    best_score = score;
                    best = Some(BestSplit {
                        feature: feat,
                        threshold,
                        score,
                    });
                }
            }
        }

        best
    }
}

/// Node-level precomputed sort by observed time.
///
/// Caches `(Y, delta)` in increasing-Y order, plus per-tied-Y-group
/// boundaries and totals. Each candidate threshold's log-rank evaluation
/// becomes a fixed-range walk over each group — known bounds let the
/// compiler autovectorize the inner accumulation.
struct SortedNode {
    /// Original sample indices in increasing-Y order.
    indices: Vec<u32>,
    /// `delta` values in increasing-Y order (`y_values` itself isn't kept
    /// — once groups are resolved, time values are no longer needed).
    delta: Vec<u8>,
    /// Half-open `[start, end)` ranges into `delta` / `left_mask`, one
    /// per tied-Y group, in increasing-Y order.
    group_range: Vec<(u32, u32)>,
    /// Total events at each group (matches `group_range`).
    group_events: Vec<u32>,
    /// Total at-risk count just before each group (matches `group_range`).
    group_at_risk: Vec<u32>,
}

impl SortedNode {
    fn from_subset(y: &[f64], delta: &[u8], sample_idx: &[u32]) -> Self {
        let n = sample_idx.len();
        let mut indices: Vec<u32> = sample_idx.to_vec();
        indices.sort_by(|&a, &b| {
            y[a as usize]
                .partial_cmp(&y[b as usize])
                .unwrap_or(Ordering::Equal)
        });
        let delta_sorted: Vec<u8> = indices.iter().map(|&i| delta[i as usize]).collect();

        // Walk sorted y values, recording group boundaries + totals.
        let mut group_range = Vec::new();
        let mut group_events = Vec::new();
        let mut group_at_risk = Vec::new();
        let mut at_risk = n as u32;
        let mut i = 0;
        while i < n {
            let t = y[indices[i] as usize];
            let start = i;
            let mut events = 0_u32;
            while i < n && y[indices[i] as usize] == t {
                events += u32::from(delta_sorted[i]);
                i += 1;
            }
            group_range.push((start as u32, i as u32));
            group_events.push(events);
            group_at_risk.push(at_risk);
            at_risk -= (i - start) as u32;
        }

        Self {
            indices,
            delta: delta_sorted,
            group_range,
            group_events,
            group_at_risk,
        }
    }

    /// Log-rank chi-square given the left-mask indexed by Y-sorted position.
    ///
    /// Same accumulation order as `_log_rank_statistic` in the pure-Python
    /// reference — float summation order is preserved so bit-exact parity
    /// holds.
    fn log_rank_score(&self, left_mask: &[u8]) -> f64 {
        debug_assert_eq!(left_mask.len(), self.delta.len());

        let mut at_risk_left = left_mask.iter().map(|&l| l as usize).sum::<usize>();
        let mut obs_minus_exp = 0.0_f64;
        let mut var_sum = 0.0_f64;

        for g in 0..self.group_range.len() {
            let (start, end) = self.group_range[g];
            let d_total = self.group_events[g];
            let at_risk_total = self.group_at_risk[g] as usize;

            let mut n_at_left = 0_u32;
            let mut d_left = 0_u32;
            let range = start as usize..end as usize;
            for (&l, &d) in left_mask[range.clone()].iter().zip(&self.delta[range]) {
                let l = l as u32;
                n_at_left += l;
                d_left += d as u32 & l;
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
            at_risk_left -= n_at_left as usize;
        }

        if var_sum <= 0.0 {
            return 0.0;
        }
        obs_minus_exp * obs_minus_exp / var_sum
    }
}

/// Variance-reduction splitting on a per-sample pseudo-outcome.
///
/// For a candidate binary split (L, R) of a node:
///
/// $$
/// \Delta(L, R) = \frac{n_L \cdot n_R}{n} \cdot (\bar Y^*_L - \bar Y^*_R)^2
/// $$
///
/// where $Y^*_i$ is the per-sample pseudo-outcome the criterion holds. The
/// score is the squared mean difference between the two children weighted
/// by their sizes — equivalent (up to a constant offset) to standard
/// MSE-based variance reduction. Higher = better.
///
/// Used as the inner split criterion for the **causal survival forest**:
/// the AIPCW pseudo-outcomes from Cui et al. (2023) are fed in as
/// `pseudo_outcome`, and a forest fit under this criterion with honesty
/// and forest-weight prediction gives the heterogeneous treatment effect
/// estimator. The criterion ignores `data.y` and `data.delta` — all
/// censoring/survival information is already absorbed into the
/// pseudo-outcome by construction.
#[derive(Debug, Clone, Copy)]
pub struct GradientCriterion<'a> {
    /// Per-sample pseudo-outcome, indexed by original training-sample
    /// index. Length must equal the training set size.
    pub pseudo_outcome: &'a [f64],
}

impl<'a> GradientCriterion<'a> {
    /// Construct with the per-sample pseudo-outcome.
    pub fn new(pseudo_outcome: &'a [f64]) -> Self {
        Self { pseudo_outcome }
    }
}

impl SplitCriterion for GradientCriterion<'_> {
    fn find_best_split(
        &self,
        data: SurvivalData<'_>,
        sample_idx: &[u32],
        candidate_features: &[u32],
        min_samples_leaf: usize,
    ) -> Option<BestSplit> {
        let n = sample_idx.len();
        if n < 2 * min_samples_leaf {
            return None;
        }

        // Total sum over the node — computed once, used to derive the
        // right-side running sum as (total - left).
        let total_sum: f64 = sample_idx
            .iter()
            .map(|&i| self.pseudo_outcome[i as usize])
            .sum();
        let n_total = n as f64;

        let mut best: Option<BestSplit> = None;
        let mut best_score = f64::NEG_INFINITY;
        let mut feat_sorted: Vec<u32> = Vec::with_capacity(n);

        for &feat in candidate_features {
            let col = data.x.column(feat as usize);
            feat_sorted.clear();
            feat_sorted.extend_from_slice(sample_idx);
            feat_sorted.sort_by(|&a, &b| {
                col[a as usize]
                    .partial_cmp(&col[b as usize])
                    .unwrap_or(Ordering::Equal)
            });

            let mut sum_left = 0.0_f64;
            let mut count_left = 0_usize;

            let mut i = 0;
            while i < n {
                let group_val = col[feat_sorted[i] as usize];
                while i < n && col[feat_sorted[i] as usize] == group_val {
                    sum_left += self.pseudo_outcome[feat_sorted[i] as usize];
                    count_left += 1;
                    i += 1;
                }
                if i == n {
                    break; // last group — no threshold after it
                }
                let next_val = col[feat_sorted[i] as usize];
                let threshold = 0.5 * (group_val + next_val);
                let count_right = n - count_left;
                if count_left < min_samples_leaf || count_right < min_samples_leaf {
                    continue;
                }
                let sum_right = total_sum - sum_left;
                let mu_l = sum_left / count_left as f64;
                let mu_r = sum_right / count_right as f64;
                let diff = mu_l - mu_r;
                let score = (count_left as f64) * (count_right as f64) / n_total * diff * diff;
                if score > best_score {
                    best_score = score;
                    best = Some(BestSplit {
                        feature: feat,
                        threshold,
                        score,
                    });
                }
            }
        }

        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;

    fn data<'a>(x: &'a Array2<f64>, y: &'a [f64], delta: &'a [u8]) -> SurvivalData<'a> {
        SurvivalData::new(x.view(), y, delta)
    }

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

    #[test]
    fn no_signal_low_score() {
        let n = 40;
        let y: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let delta = vec![1_u8; n];
        let x_data: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
        let idx: Vec<u32> = (0..n as u32).collect();
        let split = LogRankCriterion.find_best_split(data(&x, &y, &delta), &idx, &[0], 5);
        if let Some(s) = split {
            assert!(
                s.score < 5.0,
                "no-signal score unexpectedly high: {}",
                s.score
            );
        }
    }

    #[test]
    fn respects_sample_idx_subset() {
        let n = 20;
        let y: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let delta = vec![1_u8; n];
        let x_data: Vec<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }).collect();
        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();

        let idx: Vec<u32> = (0..(n / 2) as u32).collect();
        let split = LogRankCriterion.find_best_split(data(&x, &y, &delta), &idx, &[0], 1);
        assert!(
            split.is_none(),
            "single-valued feature should give no split"
        );
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
        let split = LogRankCriterion.find_best_split(data(&x, &y, &delta), &idx, &[0], 1);
        if let Some(s) = split {
            assert_relative_eq!(s.score, 0.0);
        }
    }

    // ---- GradientCriterion ----

    /// Pseudo-outcome perfectly determined by `x[:, 0] > 0.5`: criterion
    /// should split on feature 0 with threshold in the right neighborhood.
    #[test]
    fn gradient_picks_right_split_with_strong_signal() {
        let n = 40;
        let mut x = Array2::<f64>::zeros((n, 2));
        let mut pseudo = vec![0.0; n];
        for i in 0..n {
            x[(i, 0)] = i as f64 / (n - 1) as f64;
            pseudo[i] = if x[(i, 0)] > 0.5 { 1.0 } else { -1.0 };
        }
        let y = vec![1.0; n];
        let delta = vec![1_u8; n];
        let idx: Vec<u32> = (0..n as u32).collect();
        let split = GradientCriterion::new(&pseudo)
            .find_best_split(data(&x, &y, &delta), &idx, &[0, 1], 5)
            .expect("a split should exist");
        assert_eq!(split.feature, 0);
        assert!(
            split.threshold > 0.4 && split.threshold < 0.6,
            "threshold {} should be near 0.5",
            split.threshold,
        );
    }

    /// Constant pseudo-outcome → every split has score 0; best score is
    /// reported but it's zero, and `score > NEG_INF` so we get some split.
    #[test]
    fn gradient_constant_pseudo_outcome_zero_score() {
        let n = 20;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
        let pseudo = vec![2.5_f64; n];
        let y = vec![1.0; n];
        let delta = vec![1_u8; n];
        let idx: Vec<u32> = (0..n as u32).collect();
        let split =
            GradientCriterion::new(&pseudo).find_best_split(data(&x, &y, &delta), &idx, &[0], 1);
        if let Some(s) = split {
            assert_relative_eq!(s.score, 0.0);
        }
    }

    /// Constant feature column → no valid threshold → no split.
    #[test]
    fn gradient_constant_feature_no_split() {
        let n = 20;
        let x = Array2::<f64>::zeros((n, 1));
        let pseudo: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y = vec![1.0; n];
        let delta = vec![1_u8; n];
        let idx: Vec<u32> = (0..n as u32).collect();
        let split =
            GradientCriterion::new(&pseudo).find_best_split(data(&x, &y, &delta), &idx, &[0], 1);
        assert!(split.is_none());
    }

    /// Variance reduction equals the closed-form `n_L n_R / n × (μ_L − μ_R)²`.
    #[test]
    fn gradient_score_matches_closed_form() {
        let n = 10;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
        // Pseudo: first 5 = 0, last 5 = 1. Perfect split at x=4.5.
        let pseudo: Vec<f64> = (0..n).map(|i| if i < 5 { 0.0 } else { 1.0 }).collect();
        let y = vec![1.0; n];
        let delta = vec![1_u8; n];
        let idx: Vec<u32> = (0..n as u32).collect();
        let split = GradientCriterion::new(&pseudo)
            .find_best_split(data(&x, &y, &delta), &idx, &[0], 1)
            .unwrap();
        // n_L = 5, n_R = 5, mu_L = 0, mu_R = 1 → score = 5*5/10 * 1 = 2.5
        assert_relative_eq!(split.score, 2.5, max_relative = 1e-12);
        assert_relative_eq!(split.threshold, 4.5);
    }

    /// Respects `min_samples_leaf`: with leaf size 8 and n=10, no split is
    /// valid (would require leaf >= 8 on each side).
    #[test]
    fn gradient_respects_min_samples_leaf() {
        let n = 10;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
        let pseudo: Vec<f64> = (0..n).map(|i| if i < 5 { 0.0 } else { 1.0 }).collect();
        let y = vec![1.0; n];
        let delta = vec![1_u8; n];
        let idx: Vec<u32> = (0..n as u32).collect();
        let split =
            GradientCriterion::new(&pseudo).find_best_split(data(&x, &y, &delta), &idx, &[0], 8);
        assert!(split.is_none());
    }
}
