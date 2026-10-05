//! Micro and macro benchmarks for the log-rank survival tree.

use ndarray::Array2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tausurv_core::{
    LogRankCriterion, SplitCriterion, StepFunction, SurvivalData, Tree, TreeConfig,
};

fn main() {
    divan::main();
}

/// Synthetic two-group survival data with continuous covariates.
fn synth(n: usize, p: usize, seed: u64) -> (Array2<f64>, Vec<f64>, Vec<u8>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x_data: Vec<f64> = (0..n * p).map(|_| rng.random::<f64>() - 0.5).collect();
    let x = Array2::from_shape_vec((n, p), x_data).unwrap();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let signal = if x[(i, 0)] > 0.0 { 1.0 } else { 0.2 };
            -rng.random::<f64>().ln() / signal
        })
        .collect();
    // 20% censoring
    let delta: Vec<u8> = (0..n)
        .map(|_| u8::from(rng.random::<f64>() > 0.2))
        .collect();
    (x, y, delta)
}

#[divan::bench(args = [(200, 5), (1_000, 10), (5_000, 20)])]
fn fit_tree(bencher: divan::Bencher, dims: (usize, usize)) {
    let (n, p) = dims;
    let (x, y, delta) = synth(n, p, 42);
    let data = SurvivalData::new(x.view(), &y, &delta);
    let config = TreeConfig::default();
    bencher
        .with_inputs(|| 0_u64)
        .bench_values(|seed| Tree::<StepFunction>::fit(data, &LogRankCriterion, &config, seed));
}

#[divan::bench(args = [200, 1_000, 5_000])]
fn predict_cumulative_hazard(bencher: divan::Bencher, n: usize) {
    let p = 10;
    let (x, y, delta) = synth(n, p, 42);
    let data = SurvivalData::new(x.view(), &y, &delta);
    let tree = Tree::<StepFunction>::fit(data, &LogRankCriterion, &TreeConfig::default(), 0);
    let times: Vec<f64> = (1..=20).map(|i| i as f64 * 0.5).collect();
    bencher.bench(|| tree.predict_cumulative_hazard(x.view(), &times));
}

/// Cost of a single best-split search at a root-sized node.
#[divan::bench(args = [(200, 5), (1_000, 10), (5_000, 20)])]
fn find_best_split(bencher: divan::Bencher, dims: (usize, usize)) {
    let (n, p) = dims;
    let (x, y, delta) = synth(n, p, 42);
    let data = SurvivalData::new(x.view(), &y, &delta);
    let sample_idx: Vec<u32> = (0..n as u32).collect();
    let features: Vec<u32> = (0..p as u32).collect();
    bencher.bench(|| LogRankCriterion.find_best_split(data, &sample_idx, &features, 15));
}
