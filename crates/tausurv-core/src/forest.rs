//! Random survival forest: rayon-parallel ensemble of [`Tree`]s.
//!
//! Per-tree seeds are derived deterministically from a master seed in a
//! single-threaded pre-pass, so forest fits are **bit-identical across
//! thread counts**. The fit itself is then `rayon::par_iter` over the
//! seeds — trees are independent.
//!
//! Prediction averages per-tree cumulative-hazard estimates. Forest
//! weights (the grf-style sparse weighting used for asymptotic-normality
//! results) are deferred to the causal-forest layer.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::data::SurvivalData;
use crate::splits::SplitCriterion;
use crate::step::StepFunction;
use crate::tree::{LeafPayload, Tree, TreeConfig};

/// Per-tree resampling strategy.
///
/// Combinations of (with-replacement, fraction) cover both classical RSF
/// and grf-style honest forests.
#[derive(Debug, Clone, Copy)]
pub struct Bootstrap {
    /// Sample with replacement (`true`) or without (`false`).
    pub with_replacement: bool,
    /// Fraction of training samples drawn per tree; `1.0` means `n`.
    pub fraction: f64,
}

impl Default for Bootstrap {
    /// Classical RSF default: bootstrap with replacement, `n` samples.
    fn default() -> Self {
        Bootstrap {
            with_replacement: true,
            fraction: 1.0,
        }
    }
}

/// Forest-fitting configuration.
#[derive(Debug, Clone, Copy)]
pub struct ForestConfig {
    /// Number of trees in the ensemble.
    pub n_trees: usize,
    /// Configuration applied to each tree.
    pub tree: TreeConfig,
    /// Per-tree resampling strategy.
    pub bootstrap: Bootstrap,
}

impl Default for ForestConfig {
    fn default() -> Self {
        ForestConfig {
            n_trees: 100,
            tree: TreeConfig::default(),
            bootstrap: Bootstrap::default(),
        }
    }
}

/// Fitted random forest, generic over the per-leaf payload `L`.
///
/// - `Forest<StepFunction>` (default): a classical random survival forest,
///   each leaf carrying a Nelson-Aalen hazard.
/// - `Forest<()>`: a partition-only forest, predictions flow through
///   [`Forest::forest_weights`] — the foundation for the causal survival
///   forest in step 5e.
#[derive(Debug, Clone)]
pub struct Forest<L: LeafPayload = StepFunction> {
    trees: Vec<Tree<L>>,
    n_features: u32,
    /// Per-tree, per-training-sample leaf id. Shape `(n_trees, n_train)`,
    /// row-major. `u32::MAX` marks samples not in that tree's estimation
    /// set (subsample/honesty filtering).
    leaf_assignment: Array2<u32>,
    /// Per-tree, per-leaf count of estimation samples. `leaf_sizes[b][l]`
    /// is the number of training samples this tree placed in leaf `l`.
    leaf_sizes: Vec<Vec<u32>>,
}

impl<L: LeafPayload> Forest<L> {
    /// Fit a forest with the given config and master seed.
    ///
    /// Per-tree seeds are derived from `seed` in a single-threaded
    /// pre-pass, so the result is invariant to the number of rayon worker
    /// threads.
    pub fn fit<C>(data: SurvivalData<'_>, criterion: &C, config: &ForestConfig, seed: u64) -> Self
    where
        C: SplitCriterion + Sync,
    {
        let tree_seeds = derive_tree_seeds(seed, config.n_trees);
        let n_train = data.n_samples();

        let trees_and_cols: Vec<(Tree<L>, Vec<u32>)> = tree_seeds
            .par_iter()
            .map(|tree_seed| {
                let mut rng = ChaCha8Rng::from_seed(*tree_seed);
                let mut sample_idx = draw_sample(n_train, &config.bootstrap, &mut rng);
                let mut assignment = vec![u32::MAX; n_train];
                let tree = Tree::fit_subset_recording(
                    data,
                    &mut sample_idx,
                    criterion,
                    &config.tree,
                    &mut rng,
                    &mut assignment,
                );
                (tree, assignment)
            })
            .collect();

        let n_trees = trees_and_cols.len();
        let mut trees: Vec<Tree<L>> = Vec::with_capacity(n_trees);
        let mut leaf_assignment = Array2::<u32>::from_elem((n_trees, n_train), u32::MAX);
        for (b, (tree, col)) in trees_and_cols.into_iter().enumerate() {
            leaf_assignment
                .row_mut(b)
                .as_slice_mut()
                .expect("row of row-major Array2 is contiguous")
                .copy_from_slice(&col);
            trees.push(tree);
        }

        let leaf_sizes: Vec<Vec<u32>> = trees
            .iter()
            .enumerate()
            .map(|(b, tree)| {
                let mut sizes = vec![0_u32; tree.n_leaves()];
                for &lid in leaf_assignment.row(b).iter() {
                    if lid != u32::MAX {
                        sizes[lid as usize] += 1;
                    }
                }
                sizes
            })
            .collect();

        Forest {
            trees,
            n_features: data.n_features() as u32,
            leaf_assignment,
            leaf_sizes,
        }
    }

    /// Number of trees in the ensemble.
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Number of features the forest was fitted on.
    pub fn n_features(&self) -> usize {
        self.n_features as usize
    }

    /// Number of training samples the forest was fitted on.
    pub fn n_train_samples(&self) -> usize {
        self.leaf_assignment.shape()[1]
    }

    /// GRF-style forest weights at a single query point.
    ///
    /// $$
    /// w_i(x) = \frac{1}{B} \sum_b \frac{\mathbb{1}\{\mathrm{leaf}_b(x) = \mathrm{leaf}_b(x_i)\}}{|\mathrm{leaf}_b(x)|}
    /// $$
    ///
    /// Returns an `(n_train,)` array. Samples not in any tree's estimation
    /// set automatically receive zero weight (sentinel never matches).
    pub fn forest_weights(&self, x_query: ArrayView1<f64>) -> Array1<f64> {
        let n_train = self.n_train_samples();
        let mut weights = Array1::<f64>::zeros(n_train);
        for (b, tree) in self.trees.iter().enumerate() {
            let target = tree.predict_leaf(x_query);
            let size = self.leaf_sizes[b][target as usize];
            if size == 0 {
                continue;
            }
            let contrib = 1.0 / size as f64;
            let row = self.leaf_assignment.row(b);
            for (i, &lid) in row.iter().enumerate() {
                if lid == target {
                    weights[i] += contrib;
                }
            }
        }
        weights /= self.trees.len() as f64;
        weights
    }

    /// Batched forest weights: one row per query. Returns shape
    /// `(n_query, n_train)`. Parallel across queries via `rayon`.
    pub fn forest_weights_batch(&self, x_queries: ArrayView2<f64>) -> Array2<f64> {
        let n_query = x_queries.shape()[0];
        let n_train = self.n_train_samples();
        let mut out = Array2::<f64>::zeros((n_query, n_train));
        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(x_queries.axis_iter(Axis(0)).into_par_iter())
            .for_each(|(mut row_out, x_row)| {
                let w = self.forest_weights(x_row);
                row_out.assign(&w);
            });
        out
    }
}

impl Forest<StepFunction> {
    /// Tree-averaged cumulative hazard $\hat\Lambda(t | x)$ at every row of
    /// `x_query` and every time in `times`. Returns an `(n, k)` array.
    ///
    /// Only available on `Forest<StepFunction>`. Partition-only forests
    /// predict via [`Forest::forest_weights`].
    pub fn predict_cumulative_hazard(
        &self,
        x_query: ArrayView2<f64>,
        times: &[f64],
    ) -> Array2<f64> {
        let n = x_query.shape()[0];
        let k = times.len();
        let mut acc = Array2::<f64>::zeros((n, k));
        for tree in &self.trees {
            acc += &tree.predict_cumulative_hazard(x_query, times);
        }
        acc / (self.trees.len() as f64)
    }
}

/// Master → per-tree seed derivation. Single-threaded, deterministic.
fn derive_tree_seeds(master_seed: u64, n_trees: usize) -> Vec<[u8; 32]> {
    let mut master = ChaCha8Rng::seed_from_u64(master_seed);
    (0..n_trees)
        .map(|_| {
            let mut s = [0_u8; 32];
            master.fill_bytes(&mut s);
            s
        })
        .collect()
}

/// Draw a per-tree sample of training indices.
fn draw_sample(n: usize, bootstrap: &Bootstrap, rng: &mut ChaCha8Rng) -> Vec<u32> {
    assert!(
        bootstrap.fraction > 0.0 && bootstrap.fraction <= 1.0,
        "bootstrap fraction must lie in (0, 1], got {}",
        bootstrap.fraction,
    );
    let k = ((n as f64) * bootstrap.fraction).round() as usize;
    let k = k.clamp(1, n);

    if bootstrap.with_replacement {
        (0..k).map(|_| rng.random_range(0..n) as u32).collect()
    } else {
        // Fisher-Yates partial shuffle: first k slots of [0..n) become the sample.
        let mut pool: Vec<u32> = (0..n as u32).collect();
        for i in 0..k {
            let j = rng.random_range(i..n);
            pool.swap(i, j);
        }
        pool.truncate(k);
        pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::splits::LogRankCriterion;
    use crate::tree::MaxFeatures;
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
    fn fits_and_predicts() {
        let (x, y, delta) = two_group_data(200);
        let data = SurvivalData::new(x.view(), &y, &delta);
        let config = ForestConfig {
            n_trees: 20,
            tree: TreeConfig {
                min_samples_leaf: 10,
                ..TreeConfig::default()
            },
            ..ForestConfig::default()
        };
        let forest: Forest<StepFunction> = Forest::fit(data, &LogRankCriterion, &config, 42);
        assert_eq!(forest.n_trees(), 20);

        let queries = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let h = forest.predict_cumulative_hazard(queries.view(), &[5.0]);
        // Group A (x[0]=0) is the high-hazard group, B (x[0]=1) is low.
        assert!(h[[0, 0]] > h[[1, 0]], "rows: {h:?}");
    }

    #[test]
    fn deterministic_across_thread_counts() {
        let (x, y, delta) = two_group_data(200);
        let data = SurvivalData::new(x.view(), &y, &delta);
        let config = ForestConfig {
            n_trees: 16,
            tree: TreeConfig {
                min_samples_leaf: 10,
                max_features: MaxFeatures::Count(2),
                ..TreeConfig::default()
            },
            ..ForestConfig::default()
        };

        let h_a = {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap();
            pool.install(|| {
                let forest: Forest<StepFunction> = Forest::fit(data, &LogRankCriterion, &config, 7);
                forest.predict_cumulative_hazard(x.view(), &[1.0, 5.0, 12.0])
            })
        };
        let h_b = {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap();
            pool.install(|| {
                let forest: Forest<StepFunction> = Forest::fit(data, &LogRankCriterion, &config, 7);
                forest.predict_cumulative_hazard(x.view(), &[1.0, 5.0, 12.0])
            })
        };
        for (a, b) in h_a.iter().zip(h_b.iter()) {
            assert_relative_eq!(*a, *b, max_relative = 1e-15);
        }
    }

    /// Forest weights at a training point: the training sample itself
    /// should receive positive weight in every tree where it was in the
    /// estimation set.
    #[test]
    fn weights_include_self() {
        let (x, y, delta) = two_group_data(120);
        let data = SurvivalData::new(x.view(), &y, &delta);
        let forest_config = ForestConfig {
            n_trees: 5,
            tree: TreeConfig {
                min_samples_leaf: 5,
                max_features: MaxFeatures::All,
                ..TreeConfig::default()
            },
            bootstrap: Bootstrap {
                with_replacement: false,
                fraction: 1.0,
            },
        };
        let forest: Forest<StepFunction> = Forest::fit(data, &LogRankCriterion, &forest_config, 0);
        // For sample 0, query at X[0]; weight for sample 0 should be > 0.
        let w = forest.forest_weights(x.row(0));
        assert!(w[0] > 0.0, "self-weight should be positive: {}", w[0]);
        assert!(w.iter().all(|&v| v >= 0.0));
    }

    /// Sum of weights across training samples is 1.0 when every tree has a
    /// non-empty target leaf — `fraction=1.0, no honesty, max_features=All`
    /// guarantees this.
    #[test]
    fn weights_sum_to_one() {
        let (x, y, delta) = two_group_data(120);
        let data = SurvivalData::new(x.view(), &y, &delta);
        let forest_config = ForestConfig {
            n_trees: 10,
            tree: TreeConfig {
                min_samples_leaf: 5,
                max_features: MaxFeatures::All,
                ..TreeConfig::default()
            },
            bootstrap: Bootstrap {
                with_replacement: false,
                fraction: 1.0,
            },
        };
        let forest: Forest<StepFunction> = Forest::fit(data, &LogRankCriterion, &forest_config, 0);
        let query = Array2::from_shape_vec((1, 3), vec![0.5, 0.0, 0.0]).unwrap();
        let w = forest.forest_weights(query.row(0));
        let total: f64 = w.sum();
        assert_relative_eq!(total, 1.0, max_relative = 1e-12);
    }

    /// Batched weights equal stacked single-query weights.
    #[test]
    fn batched_weights_match_pointwise() {
        let (x, y, delta) = two_group_data(80);
        let data = SurvivalData::new(x.view(), &y, &delta);
        let forest_config = ForestConfig {
            n_trees: 8,
            tree: TreeConfig {
                min_samples_leaf: 5,
                ..TreeConfig::default()
            },
            ..ForestConfig::default()
        };
        let forest: Forest<StepFunction> = Forest::fit(data, &LogRankCriterion, &forest_config, 11);

        let queries =
            Array2::from_shape_vec((3, 3), vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.5])
                .unwrap();
        let batched = forest.forest_weights_batch(queries.view());
        for i in 0..3 {
            let w = forest.forest_weights(queries.row(i));
            for j in 0..forest.n_train_samples() {
                assert_relative_eq!(batched[[i, j]], w[j], max_relative = 1e-15);
            }
        }
    }

    /// A `Forest<()>` (partition-only) fits and supports `forest_weights`
    /// like the survival forest, with identical structure given the same
    /// seed. The leaf payload only changes what's stored at the leaves,
    /// not the splits chosen.
    #[test]
    fn partition_forest_matches_survival_structure() {
        let (x, y, delta) = two_group_data(120);
        let data = SurvivalData::new(x.view(), &y, &delta);
        let forest_config = ForestConfig {
            n_trees: 5,
            tree: TreeConfig {
                min_samples_leaf: 5,
                max_features: MaxFeatures::All,
                ..TreeConfig::default()
            },
            bootstrap: Bootstrap {
                with_replacement: false,
                fraction: 1.0,
            },
        };
        let survival: Forest<StepFunction> =
            Forest::fit(data, &LogRankCriterion, &forest_config, 17);
        let partition: Forest<()> = Forest::fit(data, &LogRankCriterion, &forest_config, 17);
        assert_eq!(survival.n_trees(), partition.n_trees());
        assert_eq!(survival.n_train_samples(), partition.n_train_samples());

        // Identical structure → forest weights agree exactly.
        let w_s = survival.forest_weights(x.row(0));
        let w_p = partition.forest_weights(x.row(0));
        for (a, b) in w_s.iter().zip(w_p.iter()) {
            assert_relative_eq!(*a, *b, max_relative = 1e-15);
        }
    }

    /// With `bootstrap=False, fraction=1.0, max_features=All, honesty=None`,
    /// no RNG drives any per-tree differences. Every tree is identical, so
    /// the forest prediction equals a single tree's prediction.
    #[test]
    fn no_resample_no_random_features_equals_single_tree() {
        let (x, y, delta) = two_group_data(150);
        let data = SurvivalData::new(x.view(), &y, &delta);
        let tree_config = TreeConfig {
            min_samples_leaf: 10,
            max_features: MaxFeatures::All,
            ..TreeConfig::default()
        };
        let forest_config = ForestConfig {
            n_trees: 5,
            tree: tree_config,
            bootstrap: Bootstrap {
                with_replacement: false,
                fraction: 1.0,
            },
        };
        let forest: Forest<StepFunction> = Forest::fit(data, &LogRankCriterion, &forest_config, 0);
        let single = Tree::fit(data, &LogRankCriterion, &tree_config, 0);

        let times = [1.0, 5.0, 10.0];
        let h_forest = forest.predict_cumulative_hazard(x.view(), &times);
        let h_single = single.predict_cumulative_hazard(x.view(), &times);
        for (f, s) in h_forest.iter().zip(h_single.iter()) {
            assert_relative_eq!(*f, *s, max_relative = 1e-15);
        }
    }
}
