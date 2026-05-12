//! Survival regression tree with log-rank splitting.
//!
//! Recursive partitioning over the covariate space; each leaf holds a
//! Nelson-Aalen cumulative-hazard step function fitted on the leaf's
//! samples. Nodes are stored in a flat `Vec<NodeKind>` with `u32` child
//! indices.

use ndarray::{Array2, ArrayView1, ArrayView2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::data::SurvivalData;
use crate::nelson_aalen::nelson_aalen;
use crate::splits::SplitCriterion;
use crate::step::StepFunction;

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

/// Tree-fitting configuration.
///
/// Defaults match `tausurv.trees.SurvivalTree`: `min_samples_leaf = 15`,
/// no depth cap, all features considered per split.
#[derive(Debug, Clone, Copy)]
pub struct TreeConfig {
    /// Minimum samples in any leaf. Splits creating a smaller leaf are
    /// rejected.
    pub min_samples_leaf: usize,
    /// Optional cap on recursion depth (root = 0).
    pub max_depth: Option<u32>,
    /// Feature subsampling per split.
    pub max_features: MaxFeatures,
}

impl Default for TreeConfig {
    fn default() -> Self {
        TreeConfig {
            min_samples_leaf: 15,
            max_depth: None,
            max_features: MaxFeatures::All,
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

/// Fitted survival regression tree.
#[derive(Debug, Clone)]
pub struct Tree {
    nodes: Vec<NodeKind>,
    leaf_hazards: Vec<StepFunction>,
    root: u32,
    n_features: u32,
}

impl Tree {
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

    /// Fit a tree on the given sample subset, partitioning the slice in
    /// place during the build.
    ///
    /// The forest layer derives per-tree subsamples and RNGs from a master
    /// seed and calls this entry point.
    pub fn fit_subset(
        data: SurvivalData<'_>,
        sample_idx: &mut [u32],
        criterion: &impl SplitCriterion,
        config: &TreeConfig,
        rng: &mut ChaCha8Rng,
    ) -> Self {
        let n_features = data.n_features() as u32;
        let mut builder = TreeBuilder {
            data,
            criterion,
            config,
            rng,
            nodes: Vec::new(),
            leaf_hazards: Vec::new(),
            n_features_per_split: config.max_features.resolve(n_features),
            feature_pool: (0..n_features).collect(),
        };
        let root = builder.build(sample_idx, 0);
        Tree {
            nodes: builder.nodes,
            leaf_hazards: builder.leaf_hazards,
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
        self.leaf_hazards.len()
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

    /// Cumulative hazard $\hat\Lambda(t | x)$ for every row of `x_query`
    /// at every time in `times`. Returns an `(n, k)` array.
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
            let hazard = &self.leaf_hazards[leaf_id as usize];
            for (j, &t) in times.iter().enumerate() {
                out[[i, j]] = hazard.at(t);
            }
        }
        out
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

struct TreeBuilder<'b, 'd, C: SplitCriterion> {
    data: SurvivalData<'d>,
    criterion: &'b C,
    config: &'b TreeConfig,
    rng: &'b mut ChaCha8Rng,
    nodes: Vec<NodeKind>,
    leaf_hazards: Vec<StepFunction>,
    n_features_per_split: u32,
    feature_pool: Vec<u32>,
}

impl<'b, 'd, C: SplitCriterion> TreeBuilder<'b, 'd, C> {
    fn build(&mut self, sample_idx: &mut [u32], depth: u32) -> u32 {
        if self.should_stop(sample_idx, depth) {
            return self.make_leaf(sample_idx);
        }

        let candidates = self.sample_features();
        let split = self.criterion.find_best_split(
            self.data,
            sample_idx,
            &candidates,
            self.config.min_samples_leaf,
        );
        let Some(split) = split else {
            return self.make_leaf(sample_idx);
        };

        let feature = split.feature as usize;
        let threshold = split.threshold;
        let x = self.data.x;
        let n_left =
            partition(sample_idx, |i| x[(i as usize, feature)] <= threshold);
        let (left_idx, right_idx) = sample_idx.split_at_mut(n_left);
        let left = self.build(left_idx, depth + 1);
        let right = self.build(right_idx, depth + 1);

        let id = self.nodes.len() as u32;
        self.nodes.push(NodeKind::Internal {
            feature: split.feature,
            threshold: split.threshold,
            left,
            right,
        });
        id
    }

    /// Three stop conditions match the pure-Python reference exactly:
    /// too few samples to produce two valid children, depth cap reached,
    /// or no events in the node.
    fn should_stop(&self, sample_idx: &[u32], depth: u32) -> bool {
        let too_small = sample_idx.len() < 2 * self.config.min_samples_leaf;
        let depth_hit = self.config.max_depth.is_some_and(|cap| depth >= cap);
        let no_events =
            !sample_idx.iter().any(|&i| self.data.delta[i as usize] == 1);
        too_small || depth_hit || no_events
    }

    fn make_leaf(&mut self, sample_idx: &[u32]) -> u32 {
        let hazard = nelson_aalen(self.data.y, self.data.delta, sample_idx);
        let hazard_id = self.leaf_hazards.len() as u32;
        self.leaf_hazards.push(hazard);
        let id = self.nodes.len() as u32;
        self.nodes.push(NodeKind::Leaf { hazard_id });
        id
    }

    /// Fisher-Yates partial shuffle of `feature_pool`'s first `k` slots.
    ///
    /// `feature_pool` is initialised to `0..n_features` once and never
    /// reset; subsequent calls still draw a uniform `k`-subset because
    /// Fisher-Yates on any permutation is unbiased.
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
            max_features: MaxFeatures::All,
        };
        let tree = Tree::fit(data, &LogRankCriterion, &config, 42);

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
            max_depth: None,
            max_features: MaxFeatures::All,
        };
        let tree = Tree::fit(data, &LogRankCriterion, &config, 1);
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
}
