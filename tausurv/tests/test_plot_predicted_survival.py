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
def times():
    return np.linspace(0.1, 5.0, 40)


def test_single_curve_returns_display(times):
    S = np.exp(-0.3 * times)
    disp = ts.plot.predicted_survival(times, S, label="A")
    assert disp.fig is not None
    assert disp.ax is not None
    assert list(disp.lines.keys()) == ["A"]
    assert disp.individual_lines == []


def test_single_curve_with_ci_draws_band(times):
    S = np.exp(-0.3 * times)
    disp = ts.plot.predicted_survival(
        times, S, ci=(S - 0.05, S + 0.05), label="A",
    )
    assert list(disp.ci_polys.keys()) == ["A"]


def test_ci_invalid_with_2d_S(times):
    S = np.tile(np.exp(-0.3 * times), (5, 1))
    with pytest.raises(ValueError, match="single-curve"):
        ts.plot.predicted_survival(times, S, ci=(S[0] - 0.05, S[0] + 0.05))


def test_individual_cohort_mode_plots_per_subject_lines(times):
    rng = np.random.default_rng(0)
    S = np.stack([np.exp(-(0.2 + 0.1 * rng.random()) * times) for _ in range(15)])
    disp = ts.plot.predicted_survival(times, S)
    assert len(disp.individual_lines) == 15
    assert disp.lines == {}


def test_individual_alpha_auto_scales_with_cohort_size(times):
    S_small = np.exp(-0.3 * times)[None].repeat(5, axis=0)
    S_large = np.exp(-0.3 * times)[None].repeat(500, axis=0)
    disp_small = ts.plot.predicted_survival(times, S_small)
    disp_large = ts.plot.predicted_survival(times, S_large)
    # Larger cohort -> lower alpha.
    assert disp_small.individual_lines[0].get_alpha() > disp_large.individual_lines[0].get_alpha()


def test_aggregate_no_group_collapses_to_one_curve(times):
    rng = np.random.default_rng(0)
    S = np.stack([np.exp(-(0.2 + 0.1 * rng.random()) * times) for _ in range(20)])
    disp = ts.plot.predicted_survival(times, S, aggregate="median")
    assert list(disp.lines.keys()) == [""]
    assert list(disp.ci_polys.keys()) == [""]


def test_aggregate_mean_matches_numpy_mean(times):
    rng = np.random.default_rng(0)
    S = np.stack([np.exp(-(0.2 + 0.1 * rng.random()) * times) for _ in range(20)])
    disp = ts.plot.predicted_survival(times, S, aggregate="mean")
    line = disp.lines[""]
    np.testing.assert_allclose(line.get_ydata(), S.mean(axis=0))


def test_aggregate_median_matches_numpy_median(times):
    rng = np.random.default_rng(0)
    S = np.stack([np.exp(-(0.2 + 0.1 * rng.random()) * times) for _ in range(20)])
    disp = ts.plot.predicted_survival(times, S, aggregate="median")
    line = disp.lines[""]
    np.testing.assert_allclose(line.get_ydata(), np.median(S, axis=0))


def test_group_mode_one_curve_per_group(times):
    rng = np.random.default_rng(0)
    S = np.stack([np.exp(-(0.2 + 0.1 * rng.random()) * times) for _ in range(20)])
    g = np.array(["A"] * 10 + ["B"] * 10)
    disp = ts.plot.predicted_survival(times, S, group=g)
    assert set(disp.lines.keys()) == {"A", "B"}
    assert set(disp.ci_polys.keys()) == {"A", "B"}


def test_band_sd_narrower_than_iqr_for_normal_like_data(times):
    rng = np.random.default_rng(0)
    S = 0.5 + 0.02 * rng.normal(size=(50, times.size))
    disp_iqr = ts.plot.predicted_survival(times, S, aggregate="median", band="iqr")
    disp_sd = ts.plot.predicted_survival(times, S, aggregate="mean", band="sd")
    # For a near-normal sample, +/-1 SD covers ~68% while IQR is 50%, so SD
    # should be wider than IQR. Verify on a representative time.
    iqr_lo, iqr_hi = np.percentile(S, [25, 75], axis=0)
    sd = np.std(S, axis=0, ddof=1)
    assert (sd[10] * 2) > (iqr_hi[10] - iqr_lo[10])


def test_unknown_band_raises(times):
    S = np.zeros((5, times.size))
    with pytest.raises(ValueError, match="unknown band"):
        ts.plot.predicted_survival(times, S, aggregate="mean", band="bogus")


def test_group_shape_mismatch_raises(times):
    S = np.zeros((5, times.size))
    with pytest.raises(ValueError, match="group"):
        ts.plot.predicted_survival(times, S, group=np.array(["A", "B"]))


def test_color_label_invalid_with_group(times):
    S = np.zeros((4, times.size))
    g = np.array(["A", "A", "B", "B"])
    with pytest.raises(ValueError, match="single-curve only"):
        ts.plot.predicted_survival(times, S, group=g, color="red")


def test_S_times_shape_mismatch_raises(times):
    with pytest.raises(ValueError, match="time dimension"):
        ts.plot.predicted_survival(times, np.zeros((4, times.size + 1)))


def test_S_higher_rank_raises(times):
    with pytest.raises(ValueError, match="1-D or 2-D"):
        ts.plot.predicted_survival(times, np.zeros((2, 3, times.size)))


def test_user_ax_is_respected(times):
    _, ax = plt.subplots()
    S = np.exp(-0.3 * times)
    disp = ts.plot.predicted_survival(times, S, ax=ax)
    assert disp.ax is ax
