"""Evaluation metrics for causal-survival learners.

PEHE and ATE error live in :mod:`causurv.metrics.pehe`. For purely
survival metrics (Brier, C-index, calibration) without a treatment
dimension, see :mod:`tausurv.metrics`.
"""

from causurv.metrics import pehe
