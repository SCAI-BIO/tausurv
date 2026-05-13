//! Survival regression tree with log-rank splitting.
//!
//! Recursive partitioning over the covariate space; each leaf holds a
//! Nelson-Aalen cumulative-hazard step function fitted on the leaf's
//! samples. Nodes are stored in a flat `Vec<NodeKind>` with `u32` child
//! indices.
//!
//! Optional **honest fitting** (grf style): the training subsample is
//! split in half — one half drives splits, the other drives leaf
//! estimates. See [`HonestyMode`].

use ndarray::{Array2, ArrayView1, ArrayView2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use std::fmt::Debug;

use crate::data::SurvivalData;
use crate::nelson_aalen::nelson_aalen;
use crate::splits::SplitCriterion;
use crate::step::StepFunction;

/// Per-leaf payload type for [`Tree`].
///
/// A tree-fitting algorithm produces some output at each leaf — for a
/// classical survival tree, the leaf payload is a Nelson-Aalen
/// [`StepFunction`]; for a partition-only tree (the basis for gradient /
/// causal forests), the leaf payload is `()` since predictions flow
/// through forest weights rather than per-leaf values.
///
/// Implementors decide how to compute the payload from the estimation
/// samples that fell into the leaf.
pub trait LeafPayload: Send + Sync + Clone + Debug {
    /// Compute the payload for a leaf containing `estimation` (original
    /// training-sample indices). `data` is the full training set; the
    /// implementation reads only what it needs.
    fn compute(data: SurvivalData<'_>, estimation: &[u32]) -> Self;
}

impl LeafPayload for StepFunction {
    fn compute(data: SurvivalData<'_>, estimation: &[u32]) -> Self {
        nelson_aalen(data.y, data.delta, estimation)
    }
}

impl LeafPayload for () {
    fn compute(_data: SurvivalData<'_>, _estimation: &[u32]) -> Self {}
}

/// How many features to consider at each split.
#[derive(Debug, Clone, Copy)]
pub enum MaxFeatures {
    /// All features.
    All,
    /// `floor(sqrt(p))`, with a floor of 1.
    Sqrt,
    /// An explicit count, clamped to `[1, p]`.
    Count(u32),
}

impl MaxFeatures {
    fn resolve(self, n_features: u32) -> u32 {
        let raw = match self {
            MaxFeatures::All => n_features,
            MaxFeatures::Sqrt => (n_features as f64).sqrt() as u32,
            MaxFeatures::Count(c) => c,
        };
        raw.clamp(1, n_features)
    }
}

/// Whether and how to split the training subsample into disjoint
/// splitting and estimation halves (Athey-Wager honest forests).
///
/// Honest fitting trades a small predictive-accuracy hit for asymptotically
/// unbiased leaf estimates. It is the foundation for valid confidence
/// intervals (and is required by the causal-survival-forest theory in
/// step 5 of the design doc).
#[derive(Debug, Clone, Copy)]
pub enum HonestyMode {
    /// Same samples drive splitting and estimation. Matches the classical
    /// (Ishwaran et al.) random survival forest.
    None,
    /// Shuffle the subsample, take the first `fraction` as the splitting
    /// set and the rest as the estimation set. `fraction` must be in
    /// `(0, 1)`.
    Split {
        /// Splitting-set proportion in `(0, 1)`; the rest becomes
        /// estimation. grf's default is `0.5`.
        fraction: f64,
    },
}

/// Tree-fitting configuration.
///
/// Non-honesty defaults match `tausurv.trees.SurvivalTree`:
/// `min_samples_leaf = 15`, no depth cap, all features considered.
#[derive(Debug, Clone, Copy)]
pub struct TreeConfig {
    /// Minimum samples in any leaf, applied to both splitting and
    /// estimation sets. Splits creating a smaller leaf are rejected.
    pub min_samples_leaf: usize,
    /// Optional cap on recursion depth (root = 0).
    pub max_depth: Option<u32>,
    /// Feature subsampling per split.
    pub max_features: MaxFeatures,
    /// Honest-fitting mode.
    pub honesty: HonestyMode,
}

impl Default for TreeConfig {
    fn default() -> Self {
        TreeConfig {
            min_samples_leaf: 15,
            max_depth: None,
            max_features: MaxFeatures::All,
            honesty: HonestyMode::None,
        }
    }
}

/// Internal node payload. Private — users see only [`Tree`].
#[derive(Debug, Clone)]
enum NodeKind {
    Internal {
        feature: u32,
        threshold: f64,
        left: u32,
        right: u32,
    },
    Leaf {
        hazard_id: u32,
    },
}

/// Fitted regression tree, generic over the per-leaf payload `L`.
///
/// - `Tree<StepFunction>` (default): leaves carry Nelson-Aalen cumulative
///   hazards — the classical survival tree.
/// - `Tree<()>`: leaves carry nothing — a partition-only tree, the basis
///   for gradient/causal forests that predict via forest weights instead
///   of per-leaf payloads.
#[derive(Debug, Clone)]
pub struct Tree<L: LeafPayload = StepFunction> {
    nodes: Vec<NodeKind>,
    leaf_payloads: Vec<L>,
    root: u32,
    n_features: u32,
}

impl<L: LeafPayload> Tree<L> {
    /// Fit a tree on the whole training set using a freshly seeded RNG.
    ///
    /// Convenience over [`Tree::fit_subset`]. For forest fitting, derive
    /// a per-tree seed and use [`Tree::fit_subset`] directly.
    pub fn fit(
        data: SurvivalData<'_>,
        criterion: &impl SplitCriterion,
        config: &TreeConfig,
        seed: u64,
    ) -> Self {
        let mut sample_idx: Vec<u32> = (0..data.n_samples() as u32).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        Tree::fit_subset(data, &mut sample_idx, criterion, config, &mut rng)
    }

    /// Fit a tree on the given sample subset.
    ///
    /// `sample_idx` is partitioned in place. In honest mode it is also
    /// shuffled up front so the first `fraction * n` indices become the
    /// splitting set and the rest the estimation set.
    pub fn fit_subset(
        data: SurvivalData<'_>,
        sample_idx: &mut [u32],
        criterion: &impl SplitCriterion,
        config: &TreeConfig,
        rng: &mut ChaCha8Rng,
    ) -> Self {
        Tree::fit_subset_inner(data, sample_idx, criterion, config, rng, None)
    }

    /// Like [`Tree::fit_subset`] but also records each estimation-set
    /// sample's final leaf id into `leaf_assignment_out` (which must have
    /// length `data.n_samples()`). Entries for samples not in the
    /// estimation set are left untouched — callers should pre-fill with
    /// `u32::MAX` to mark them.
    ///
    /// Used by [`crate::Forest`] to compute forest weights.
    pub fn fit_subset_recording(
        data: SurvivalData<'_>,
        sample_idx: &mut [u32],
        criterion: &impl SplitCriterion,
        config: &TreeConfig,
        rng: &mut ChaCha8Rng,
        leaf_assignment_out: &mut [u32],
    ) -> Self {
        debug_assert_eq!(leaf_assignment_out.len(), data.n_samples());
        Tree::fit_subset_inner(
            data,
            sample_idx,
            criterion,
            config,
            rng,
            Some(leaf_assignment_out),
        )
    }

    fn fit_subset_inner(
        data: SurvivalData<'_>,
        sample_idx: &mut [u32],
        criterion: &impl SplitCriterion,
        config: &TreeConfig,
        rng: &mut ChaCha8Rng,
        leaf_assignment_out: Option<&mut [u32]>,
    ) -> Self {
        let n_features = data.n_features() as u32;
        let n_total = sample_idx.len();

        let (n_split, mut estimation) = match config.honesty {
            HonestyMode::None => (n_total, sample_idx.to_vec()),
            HonestyMode::Split { fraction } => {
                assert!(
                    fraction > 0.0 && fraction < 1.0,
                    "honesty fraction must lie in (0, 1), got {fraction}",
                );
                shuffle_in_place(sample_idx, rng);
                let raw = ((n_total as f64) * fraction).round() as usize;
                let n_split = raw.clamp(1, n_total.saturating_sub(1));
                let estimation = sample_idx[n_split..].to_vec();
                (n_split, estimation)
            }
        };
        let splitting = &mut sample_idx[..n_split];

        let mut builder = TreeBuilder::<L, _> {
            data,
            criterion,
            config,
            rng,
            nodes: Vec::new(),
            leaf_payloads: Vec::<L>::new(),
            n_features_per_split: config.max_features.resolve(n_features),
            feature_pool: (0..n_features).collect(),
            leaf_assignment_out,
        };
        let root = builder.build(splitting, &mut estimation, 0);
        Tree {
            nodes: builder.nodes,
            leaf_payloads: builder.leaf_payloads,
            root,
            n_features,
        }
    }

    /// Number of features the tree was fitted on.
    pub fn n_features(&self) -> usize {
        self.n_features as usize
    }

    /// Total node count (internal + leaf).
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of leaves.
    pub fn n_leaves(&self) -> usize {
        self.leaf_payloads.len()
    }

    /// Maximum root-to-leaf depth.
    pub fn depth(&self) -> u32 {
        self.depth_of(self.root)
    }

    /// Index of the leaf reached by `x`.
    pub fn predict_leaf(&self, x: ArrayView1<f64>) -> u32 {
        let mut node = self.root;
        loop {
            match &self.nodes[node as usize] {
                NodeKind::Leaf { hazard_id } => return *hazard_id,
                NodeKind::Internal { feature, threshold, left, right } => {
                    node = if x[*feature as usize] <= *threshold {
                        *left
                    } else {
                        *right
                    };
                }
            }
        }
    }

    fn depth_of(&self, node: u32) -> u32 {
        match &self.nodes[node as usize] {
            NodeKind::Leaf { .. } => 0,
            NodeKind::Internal { left, right, .. } => {
                1 + self.depth_of(*left).max(self.depth_of(*right))
            }
        }
    }
}

impl Tree<StepFunction> {
    /// Cumulative hazard $\hat\Lambda(t | x)$ for every row of `x_query`
    /// at every time in `times`. Returns an `(n, k)` array.
    ///
    /// Only available on `Tree<StepFunction>` — partition-only trees
    /// (`Tree<()>`) don't carry hazards and predict via forest weights.
    pub fn predict_cumulative_hazard(
        &self,
        x_query: ArrayView2<f64>,
        times: &[f64],
    ) -> Array2<f64> {
        let n = x_query.shape()[0];
        let k = times.len();
        let mut out = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let leaf_id = self.predict_leaf(x_query.row(i));
            let hazard = &self.leaf_payloads[leaf_id as usize];
            for (j, &t) in times.iter().enumerate() {
                out[[i, j]] = hazard.at(t);
            }
        }
        out
    }
}

struct TreeBuilder<'b, 'd, L: LeafPayload, C: SplitCriterion> {
    data: SurvivalData<'d>,
    criterion: &'b C,
    config: &'b TreeConfig,
    rng: &'b mut ChaCha8Rng,
    nodes: Vec<NodeKind>,
    leaf_payloads: Vec<L>,
    n_features_per_split: u32,
    feature_pool: Vec<u32>,
    /// If `Some`, each estimation-set sample's leaf id is written to
    /// `leaf_assignment_out[original_index]` when its leaf is created.
    leaf_assignment_out: Option<&'b mut [u32]>,
}

impl<'b, 'd, L: LeafPayload, C: SplitCriterion> TreeBuilder<'b, 'd, L, C> {
    fn build(
        &mut self,
        splitting: &mut [u32],
        estimation: &mut [u32],
        depth: u32,
    ) -> u32 {
        if self.should_stop(splitting, estimation, depth) {
            return self.make_leaf(estimation);
        }

        let candidates = self.sample_features();
        let split = self.criterion.find_best_split(
            self.data,
            splitting,
            &candidates,
            self.config.min_samples_leaf,
        );
        let Some(split) = split else {
            return self.make_leaf(estimation);
        };

        let feature = split.feature as usize;
        let threshold = split.threshold;
        let x = self.data.x;
        let pred = |i: u32| x[(i as usize, feature)] <= threshold;
        let n_left_split = partition(splitting, pred);
        let n_left_est = partition(estimation, pred);

        // Estimation-side leaf-size constraint: forcibly leaf if violated.
        // Mirrors grf's `min.node.size` applied symmetrically to both sets.
        let min = self.config.min_samples_leaf;
        if n_left_est < min || estimation.len() - n_left_est < min {
            return self.make_leaf(estimation);
        }

        let (s_l, s_r) = splitting.split_at_mut(n_left_split);
        let (e_l, e_r) = estimation.split_at_mut(n_left_est);
        let left = self.build(s_l, e_l, depth + 1);
        let right = self.build(s_r, e_r, depth + 1);

        let id = self.nodes.len() as u32;
        self.nodes.push(NodeKind::Internal {
            feature: split.feature,
            threshold: split.threshold,
            left,
            right,
        });
        id
    }

    /// Generic stop conditions:
    ///   - splitting set too small to produce two valid children
    ///   - estimation set empty (no samples to feed the leaf)
    ///   - depth cap reached
    ///
    /// Criterion-specific "nothing to split on" cases (e.g. all-censored
    /// nodes for log-rank) are handled inside the criterion's
    /// `find_best_split` returning `None`.
    fn should_stop(
        &self,
        splitting: &[u32],
        estimation: &[u32],
        depth: u32,
    ) -> bool {
        let too_small =
            splitting.len() < 2 * self.config.min_samples_leaf || estimation.is_empty();
        let depth_hit = self.config.max_depth.is_some_and(|cap| depth >= cap);
        too_small || depth_hit
    }

    fn make_leaf(&mut self, estimation: &[u32]) -> u32 {
        let payload = L::compute(self.data, estimation);
        let hazard_id = self.leaf_payloads.len() as u32;
        self.leaf_payloads.push(payload);
        let id = self.nodes.len() as u32;
        if let Some(out) = self.leaf_assignment_out.as_deref_mut() {
            for &orig in estimation {
                out[orig as usize] = hazard_id;
            }
        }
        self.nodes.push(NodeKind::Leaf { hazard_id });
        id
    }

    /// Fisher-Yates partial shuffle of `feature_pool`'s first `k` slots.
    fn sample_features(&mut self) -> Vec<u32> {
        let n = self.feature_pool.len();
        let k = self.n_features_per_split as usize;
        if k >= n {
            return self.feature_pool.clone();
        }
        for i in 0..k {
            let j = self.rng.random_range(i..n);
            self.feature_pool.swap(i, j);
        }
        self.feature_pool[..k].to_vec()
    }
}

/// Fisher-Yates shuffle, in place.
fn shuffle_in_place(slice: &mut [u32], rng: &mut ChaCha8Rng) {
    for i in (1..slice.len()).rev() {
        let j = rng.random_range(0..=i);
        slice.swap(i, j);
    }
}

/// Lomuto-style in-place partition: elements satisfying `pred` come
/// first. Returns the count of matching elements.
fn partition<F: Fn(u32) -> bool>(idx: &mut [u32], pred: F) -> usize {
    let mut i = 0;
    let mut j = idx.len();
    while i < j {
        if pred(idx[i]) {
            i += 1;
        } else {
            j -= 1;
            idx.swap(i, j);
        }
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::splits::LogRankCriterion;
    use approx::assert_relative_eq;
    use ndarray::Array2;

    fn two_group_data(n: usize) -> (Array2<f64>, Vec<f64>, Vec<u8>) {
        let mut y = vec![0.0; n];
        let delta = vec![1_u8; n];
        let mut x = Array2::<f64>::zeros((n, 3));
        for i in 0..n / 2 {
            y[i] = 1.0 + (i as f64) * 0.01;
            x[(i, 0)] = 0.0;
        }
        for i in n / 2..n {
            y[i] = 10.0 + (i as f64) * 0.01;
            x[(i, 0)] = 1.0;
        }
        (x, y, delta)
    }

    #[test]
    fn fits_obvious_signal() {
        let (x, y, delta) = two_group_data(200);
        let data = SurvivalData::new(x.view(), &y, &delta);
        let config = TreeConfig {
            min_samples_leaf: 10,
            max_depth: Some(2),
            ..TreeConfig::default()
        };
        let tree: Tree<StepFunction> =
            Tree::fit(data, &LogRankCriterion, &config, 42);

        assert!(tree.n_leaves() >= 2);
        match &tree.nodes[tree.root as usize] {
            NodeKind::Internal { feature, threshold, .. } => {
                assert_eq!(*feature, 0);
                assert!(*threshold > 0.0 && *threshold < 1.0);
            }
            _ => panic!("root should be internal for this signal"),
        }
    }

    #[test]
    fn predictions_reflect_separation() {
        let n = 200;
        let mut y = vec![0.0; n];
        let delta = vec![1_u8; n];
        let mut x = Array2::<f64>::zeros((n, 1));
        for i in 0..n / 2 {
            y[i] = 1.0 + (i as f64) * 0.01;
            x[(i, 0)] = 0.0;
        }
        for i in n / 2..n {
            y[i] = 10.0 + (i as f64) * 0.01;
            x[(i, 0)] = 1.0;
        }
        let data = SurvivalData::new(x.view(), &y, &delta);
        let tree = Tree::fit(data, &LogRankCriterion, &TreeConfig::default(), 7);

        let queries = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let times = [5.0, 12.0];
        let h = tree.predict_cumulative_hazard(queries.view(), &times);
        assert!(h[[0, 0]] > h[[1, 0]], "rows: {h:?}");
        assert!(h[[1, 1]] > 0.0);
    }

    #[test]
    fn stops_when_no_events() {
        let n = 30;
        let y: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let delta = vec![0_u8; n];
        let x = Array2::<f64>::zeros((n, 2));
        let data = SurvivalData::new(x.view(), &y, &delta);
        let config = TreeConfig {
            min_samples_leaf: 5,
            ..TreeConfig::default()
        };
        let tree: Tree<StepFunction> =
            Tree::fit(data, &LogRankCriterion, &config, 1);
        assert_eq!(tree.n_leaves(), 1);
        assert_eq!(tree.n_nodes(), 1);
        assert_eq!(tree.depth(), 0);
    }

    #[test]
    fn deterministic_under_seed() {
        let n = 100;
        let y: Vec<f64> = (0..n).map(|i| ((i * 7) % 19 + 1) as f64).collect();
        let delta: Vec<u8> = (0..n).map(|i| ((i % 3) != 0) as u8).collect();
        let x_data: Vec<f64> = (0..n * 4).map(|k| ((k * 13) % 17) as f64).collect();
        let x = Array2::from_shape_vec((n, 4), x_data).unwrap();
        let data = SurvivalData::new(x.view(), &y, &delta);
        let config = TreeConfig {
            min_samples_leaf: 5,
            max_depth: Some(4),
            max_features: MaxFeatures::Count(2),
            ..TreeConfig::default()
        };

        let tree_a = Tree::fit(data, &LogRankCriterion, &config, 123);
        let tree_b = Tree::fit(data, &LogRankCriterion, &config, 123);

        assert_eq!(tree_a.n_nodes(), tree_b.n_nodes());
        assert_eq!(tree_a.n_leaves(), tree_b.n_leaves());

        let times = [1.0, 5.0, 10.0, 20.0];
        let h_a = tree_a.predict_cumulative_hazard(x.view(), &times);
        let h_b = tree_b.predict_cumulative_hazard(x.view(), &times);
        for (a, b) in h_a.iter().zip(h_b.iter()) {
            assert_relative_eq!(*a, *b, max_relative = 1e-15);
        }
    }

    #[test]
    fn honest_mode_fits() {
        let (x, y, delta) = two_group_data(400);
        let data = SurvivalData::new(x.view(), &y, &delta);
        let config = TreeConfig {
            min_samples_leaf: 15,
            honesty: HonestyMode::Split { fraction: 0.5 },
            ..TreeConfig::default()
        };
        let tree = Tree::fit(data, &LogRankCriterion, &config, 42);
        assert!(tree.n_leaves() >= 2);
        // Leaf hazards should still distinguish the two groups even after
        // honesty halves the estimation set.
        let queries = Array2::from_shape_vec((2, 3), vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
        ]).unwrap();
        let h = tree.predict_cumulative_hazard(queries.view(), &[5.0]);
        assert!(h[[0, 0]] > h[[1, 0]], "honest tree should still separate groups: {h:?}");
    }

    #[test]
    fn honest_mode_deterministic_under_seed() {
        let (x, y, delta) = two_group_data(300);
        let data = SurvivalData::new(x.view(), &y, &delta);
        let config = TreeConfig {
            min_samples_leaf: 10,
            honesty: HonestyMode::Split { fraction: 0.5 },
            ..TreeConfig::default()
        };
        let tree_a = Tree::fit(data, &LogRankCriterion, &config, 99);
        let tree_b = Tree::fit(data, &LogRankCriterion, &config, 99);
        let times = [1.0, 5.0, 12.0];
        let h_a = tree_a.predict_cumulative_hazard(x.view(), &times);
        let h_b = tree_b.predict_cumulative_hazard(x.view(), &times);
        for (a, b) in h_a.iter().zip(h_b.iter()) {
            assert_relative_eq!(*a, *b, max_relative = 1e-15);
        }
    }

    /// A `Tree<()>` and a `Tree<StepFunction>` fit on the same data with
    /// the same seed must have the same structure — the leaf payload type
    /// only changes what's stored at the leaves, not where the splits
    /// land.
    #[test]
    fn partition_tree_has_same_structure_as_survival_tree() {
        let (x, y, delta) = two_group_data(200);
        let data = SurvivalData::new(x.view(), &y, &delta);
        let survival: Tree<StepFunction> =
            Tree::fit(data, &LogRankCriterion, &TreeConfig::default(), 0);
        let partition: Tree<()> =
            Tree::fit(data, &LogRankCriterion, &TreeConfig::default(), 0);
        assert_eq!(survival.n_nodes(), partition.n_nodes());
        assert_eq!(survival.n_leaves(), partition.n_leaves());
        assert_eq!(survival.depth(), partition.depth());
        // `predict_cumulative_hazard` only compiles on Tree<StepFunction> —
        // attempting it on the partition tree would be a type error.
        let _ = survival.predict_cumulative_hazard(x.view(), &[1.0, 5.0]);
    }

    #[test]
    fn honest_differs_from_non_honest() {
        let (x, y, delta) = two_group_data(300);
        let data = SurvivalData::new(x.view(), &y, &delta);
        let base = TreeConfig {
            min_samples_leaf: 10,
            ..TreeConfig::default()
        };
        let tree_non = Tree::fit(data, &LogRankCriterion, &base, 7);
        let honest = TreeConfig {
            honesty: HonestyMode::Split { fraction: 0.5 },
            ..base
        };
        let tree_hon = Tree::fit(data, &LogRankCriterion, &honest, 7);
        let times = [5.0, 10.0];
        let h_non = tree_non.predict_cumulative_hazard(x.view(), &times);
        let h_hon = tree_hon.predict_cumulative_hazard(x.view(), &times);
        // Hazard estimates should differ — honest uses a different (half)
        // sample for Nelson-Aalen, so leaf values are not identical.
        let max_diff = h_non
            .iter()
            .zip(h_hon.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff > 1e-6,
            "honest and non-honest produced identical predictions (max_diff={max_diff})",
        );
    }
}
