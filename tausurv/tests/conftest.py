from __future__ import annotations

import os

# Under pytest-xdist every worker is its own process. Uncapped, each would
# start a BLAS, PyTorch and Rayon thread per core, and the oversubscribed
# machine runs slower than a single process. Set before numpy is imported.
if "PYTEST_XDIST_WORKER" in os.environ:
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.network (hit external URLs)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="needs --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def network_cache(tmp_path_factory):
    """Session-scoped cache dir shared across all @network tests."""
    return tmp_path_factory.mktemp("tausurv_datasets")


@pytest.fixture
def random_survival_data():
    rng = np.random.default_rng(42)
    n = 200
    risk_score = rng.normal(size=n)
    # higher risk -> shorter expected event time
    event_time = rng.exponential(scale=np.exp(-risk_score))
    censoring_time = rng.exponential(scale=2.0, size=n)
    event_indicator = (event_time <= censoring_time).astype(np.int8)
    observed_time = np.minimum(event_time, censoring_time)
    return observed_time, event_indicator, risk_score
