//! Performance-critical survival-analysis primitives.
//!
//! This crate is intentionally Python-agnostic. All PyO3 bindings live in
//! the sibling `tausurv-pyo3` crate.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![cfg_attr(test, allow(missing_docs))]

pub mod data;
pub mod nelson_aalen;
pub mod splits;
pub mod step;
pub mod tree;

pub use data::SurvivalData;
pub use nelson_aalen::nelson_aalen;
pub use splits::{BestSplit, LogRankCriterion, SplitCriterion};
pub use step::{Side, StepFunction};
pub use tree::{MaxFeatures, Tree, TreeConfig};

/// Crate version, surfaced through the Python binding for sanity checks.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
