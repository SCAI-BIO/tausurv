//! Flat, language-neutral form of fitted trees and forests.
//!
//! A fitted tree is a node table plus one payload per leaf; a forest is
//! its trees' tables concatenated, with offsets, plus the leaf each
//! training sample fell into in each tree. [`crate::Tree::to_arrays`] and
//! [`crate::Forest::to_arrays`] export this form; the `from_arrays`
//! constructors validate every invariant on the way back, so a file that
//! was edited or written by another version fails loudly instead of
//! producing wrong predictions.
//!
//! The layout follows scikit-learn's tree arrays: one column per node
//! attribute, children by index, and CSR-style offsets for the ragged
//! leaf grids.

use std::ops::Range;

use thiserror::Error;

/// A fitted tree or forest could not be rebuilt from its arrays.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[error("invalid tree arrays: {0}")]
pub struct InvalidArrays(pub String);

/// Node table of one or more trees, one entry per node.
#[derive(Debug, Clone, Default)]
pub struct NodeArrays {
    /// Split feature, or `-1` for a leaf.
    pub feature: Vec<i64>,
    /// Split threshold; `x[feature] <= threshold` goes left. `NaN` for a leaf.
    pub threshold: Vec<f64>,
    /// Left child, as an index within the tree; `u32::MAX` for a leaf.
    pub left: Vec<u32>,
    /// Right child, as an index within the tree; `u32::MAX` for a leaf.
    pub right: Vec<u32>,
    /// Leaf payload index within the tree; `u32::MAX` for an internal node.
    pub leaf: Vec<u32>,
}

impl NodeArrays {
    /// Number of nodes, once the columns are known to agree in length.
    pub fn len(&self) -> usize {
        self.feature.len()
    }

    /// `true` when the table holds no node.
    pub fn is_empty(&self) -> bool {
        self.feature.is_empty()
    }

    pub(crate) fn check_lengths(&self) -> Result<(), InvalidArrays> {
        let n = self.feature.len();
        if [
            self.threshold.len(),
            self.left.len(),
            self.right.len(),
            self.leaf.len(),
        ]
        .iter()
        .any(|&len| len != n)
        {
            return Err(InvalidArrays("node columns differ in length".to_string()));
        }
        Ok(())
    }

    pub(crate) fn push_internal(&mut self, feature: u32, threshold: f64, left: u32, right: u32) {
        self.feature.push(i64::from(feature));
        self.threshold.push(threshold);
        self.left.push(left);
        self.right.push(right);
        self.leaf.push(u32::MAX);
    }

    pub(crate) fn push_leaf(&mut self, leaf: u32) {
        self.feature.push(-1);
        self.threshold.push(f64::NAN);
        self.left.push(u32::MAX);
        self.right.push(u32::MAX);
        self.leaf.push(leaf);
    }

    pub(crate) fn extend(&mut self, other: &NodeArrays) {
        self.feature.extend_from_slice(&other.feature);
        self.threshold.extend_from_slice(&other.threshold);
        self.left.extend_from_slice(&other.left);
        self.right.extend_from_slice(&other.right);
        self.leaf.extend_from_slice(&other.leaf);
    }

    pub(crate) fn slice(&self, range: Range<usize>) -> NodeArrays {
        NodeArrays {
            feature: self.feature[range.clone()].to_vec(),
            threshold: self.threshold[range.clone()].to_vec(),
            left: self.left[range.clone()].to_vec(),
            right: self.right[range.clone()].to_vec(),
            leaf: self.leaf[range].to_vec(),
        }
    }
}

/// Step-function payloads of one or more trees, concatenated.
///
/// Leaf `l` owns the grid points `offsets[l]..offsets[l + 1]` of `time`
/// and `value`. Payload-free trees (`Tree<()>`) leave every column empty.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LeafArrays {
    /// Start of each leaf's grid; length `n_leaves + 1`, first entry `0`.
    pub offsets: Vec<u64>,
    /// Grid times of all leaves, each leaf's strictly increasing.
    pub time: Vec<f64>,
    /// Grid values of all leaves.
    pub value: Vec<f64>,
    /// Value below each leaf's first grid time.
    pub baseline: Vec<f64>,
    /// Continuity per leaf: `0` right-continuous, `1` left-continuous.
    pub side: Vec<u8>,
}

impl LeafArrays {
    pub(crate) fn extend(&mut self, other: &LeafArrays) {
        if self.offsets.is_empty() {
            self.offsets.push(0);
        }
        let base = self.time.len() as u64;
        self.offsets
            .extend(other.offsets.iter().skip(1).map(|&o| o + base));
        self.time.extend_from_slice(&other.time);
        self.value.extend_from_slice(&other.value);
        self.baseline.extend_from_slice(&other.baseline);
        self.side.extend_from_slice(&other.side);
    }

    /// The payloads of leaves `range`, rebased so their offsets start at 0.
    pub(crate) fn slice(&self, range: Range<usize>) -> Result<LeafArrays, InvalidArrays> {
        if self.offsets.is_empty() {
            return Ok(LeafArrays::default());
        }
        if range.end >= self.offsets.len() {
            return Err(InvalidArrays(format!(
                "leaf range {range:?} exceeds the {} leaves present",
                self.offsets.len() - 1
            )));
        }
        let start = self.offsets[range.start] as usize;
        let end = self.offsets[range.end] as usize;
        if start > end || end > self.time.len() {
            return Err(InvalidArrays("leaf offsets are not monotone".to_string()));
        }
        Ok(LeafArrays {
            offsets: self.offsets[range.start..=range.end]
                .iter()
                .map(|&o| o - start as u64)
                .collect(),
            time: self.time[start..end].to_vec(),
            value: self.value[start..end].to_vec(),
            baseline: self.baseline[range.clone()].to_vec(),
            side: self.side[range].to_vec(),
        })
    }
}

/// One fitted tree in flat form.
#[derive(Debug, Clone)]
pub struct TreeArrays {
    /// Number of features the tree was fitted on.
    pub n_features: u32,
    /// Index of the root node.
    pub root: u32,
    /// The node table.
    pub nodes: NodeArrays,
    /// The leaf payloads.
    pub leaves: LeafArrays,
}

/// One fitted forest in flat form: its trees' tables concatenated.
#[derive(Debug, Clone)]
pub struct ForestArrays {
    /// Number of features the forest was fitted on.
    pub n_features: u32,
    /// Number of training samples the forest was fitted on.
    pub n_train: u32,
    /// Node range of tree `b` is `node_offsets[b]..node_offsets[b + 1]`;
    /// length `n_trees + 1`.
    pub node_offsets: Vec<u64>,
    /// Leaf range of tree `b`, in the same form; length `n_trees + 1`.
    pub leaf_offsets: Vec<u64>,
    /// Root node of each tree, relative to the tree's node range.
    pub root: Vec<u32>,
    /// All trees' nodes, tree by tree.
    pub nodes: NodeArrays,
    /// All trees' leaf payloads, tree by tree.
    pub leaves: LeafArrays,
    /// Row-major `(n_trees, n_train)`: the leaf each training sample fell
    /// into in each tree, `u32::MAX` when the sample was not in that
    /// tree's estimation set.
    pub leaf_assignment: Vec<u32>,
}

impl NodeArrays {
    /// `true` when both tables describe the same nodes. Leaf thresholds
    /// are `NaN`, so this is not `==`.
    pub fn same_as(&self, other: &NodeArrays) -> bool {
        self.feature == other.feature
            && self.left == other.left
            && self.right == other.right
            && self.leaf == other.leaf
            && self.threshold.len() == other.threshold.len()
            && self
                .threshold
                .iter()
                .zip(&other.threshold)
                .all(|(a, b)| a == b || (a.is_nan() && b.is_nan()))
    }
}

/// Check that `offsets` describes `n_items` consecutive ranges over
/// `total` elements: starts at 0, never decreases, ends at `total`.
pub(crate) fn check_offsets(
    offsets: &[u64],
    n_items: usize,
    total: usize,
    what: &str,
) -> Result<(), InvalidArrays> {
    if offsets.len() != n_items + 1 {
        return Err(InvalidArrays(format!(
            "{what} offsets have length {} but {n_items} items need {}",
            offsets.len(),
            n_items + 1
        )));
    }
    if offsets[0] != 0 || offsets.windows(2).any(|w| w[0] > w[1]) {
        return Err(InvalidArrays(format!(
            "{what} offsets must start at 0 and never decrease"
        )));
    }
    if offsets[n_items] != total as u64 {
        return Err(InvalidArrays(format!(
            "{what} offsets end at {} but there are {total} elements",
            offsets[n_items]
        )));
    }
    Ok(())
}
