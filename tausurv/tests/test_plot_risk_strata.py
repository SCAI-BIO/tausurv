from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import tausurv as ts  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def risk_data():
    rng = np.random.default_rng(0)
    n = 200
    risk = rng.normal(0, 1, n)
    T = rng.exponential(np.exp(-0.5 * risk), n)
    C = rng.exponential(4.0, n)
    Y = np.minimum(T, C)
    D = (T <= C).astype(int)
    return Y, D, risk


def test_returns_display(risk_data):
    Y, D, R = risk_data
    disp = ts.plot.risk_strata(Y, D, R, n_bins=4)
    assert disp.fig is not None
    assert disp.ax is not None
    assert disp.at_risk_ax is not None
    assert disp.bin_edges.shape == (5,)
    assert disp.n_per_bin.shape == (4,)
    assert len(disp.bin_labels) == 4


def test_lines_ordered_lowest_to_highest_risk(risk_data):
    Y, D, R = risk_data
    disp = ts.plot.risk_strata(Y, D, R, n_bins=4)
    assert list(disp.lines.keys()) == ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]


def test_quartile_bins_balanced(risk_data):
    Y, D, R = risk_data
    disp = ts.plot.risk_strata(Y, D, R, n_bins=4)
    assert disp.n_per_bin.sum() == len(Y)
    # Equal-quantile binning -> within +/-1 of balanced.
    assert disp.n_per_bin.max() - disp.n_per_bin.min() <= 1


def test_decile_labels(risk_data):
    Y, D, R = risk_data
    disp = ts.plot.risk_strata(Y, D, R, n_bins=10)
    assert disp.bin_labels == [f"D{i}" for i in range(1, 11)]


def test_custom_labels(risk_data):
    Y, D, R = risk_data
    disp = ts.plot.risk_strata(
        Y,
        D,
        R,
        n_bins=3,
        labels=["Low", "Medium", "High"],
    )
    assert disp.bin_labels == ["Low", "Medium", "High"]
    assert list(disp.lines.keys()) == ["Low", "Medium", "High"]


def test_custom_labels_wrong_length_raises(risk_data):
    Y, D, R = risk_data
    with pytest.raises(ValueError, match="length n_bins"):
        ts.plot.risk_strata(Y, D, R, n_bins=4, labels=["A", "B"])


def test_equal_width_binning(risk_data):
    Y, D, R = risk_data
    disp = ts.plot.risk_strata(Y, D, R, n_bins=3, binning="equal_width")
    assert disp.bin_edges[0] == pytest.approx(R.min())
    assert disp.bin_edges[-1] == pytest.approx(R.max())
    # Equal-width: differences between adjacent edges are equal.
    np.testing.assert_allclose(np.diff(disp.bin_edges), np.diff(disp.bin_edges)[0])


def test_manual_binning(risk_data):
    Y, D, R = risk_data
    disp = ts.plot.risk_strata(
        Y,
        D,
        R,
        n_bins=3,
        binning="manual",
        breakpoints=[-0.5, 0.5],
    )
    assert disp.bin_edges[1] == pytest.approx(-0.5)
    assert disp.bin_edges[2] == pytest.approx(0.5)


def test_manual_binning_without_breakpoints_raises(risk_data):
    Y, D, R = risk_data
    with pytest.raises(ValueError, match="requires breakpoints"):
        ts.plot.risk_strata(Y, D, R, n_bins=3, binning="manual")


def test_manual_binning_wrong_breakpoint_count_raises(risk_data):
    Y, D, R = risk_data
    with pytest.raises(ValueError, match="length n_bins - 1"):
        ts.plot.risk_strata(
            Y,
            D,
            R,
            n_bins=3,
            binning="manual",
            breakpoints=[-0.5],
        )


def test_unknown_binning_raises(risk_data):
    Y, D, R = risk_data
    with pytest.raises(ValueError, match="unknown binning"):
        ts.plot.risk_strata(Y, D, R, n_bins=3, binning="bogus")


def test_too_few_bins_raises(risk_data):
    Y, D, R = risk_data
    with pytest.raises(ValueError, match="at least 2"):
        ts.plot.risk_strata(Y, D, R, n_bins=1)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        ts.plot.risk_strata([1.0, 2.0], [1, 1], [0.1])


def test_at_risk_disabled(risk_data):
    Y, D, R = risk_data
    disp = ts.plot.risk_strata(Y, D, R, n_bins=3, at_risk=False)
    assert disp.at_risk_ax is None


def test_ci_disabled(risk_data):
    Y, D, R = risk_data
    disp = ts.plot.risk_strata(Y, D, R, n_bins=3, ci=False)
    assert disp.ci_polys == {}


def test_title_applied(risk_data):
    Y, D, R = risk_data
    disp = ts.plot.risk_strata(Y, D, R, n_bins=3, title="My title")
    assert disp.ax.get_title() == "My title"
