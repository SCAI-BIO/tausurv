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
def grid():
    return np.linspace(0.5, 5.0, 12)


@pytest.fixture
def models(grid):
    return {
        "Cox":     {"times": grid, "values": 0.74 + 0.02 * np.sin(grid),
                    "ci": (0.71 + 0.02 * np.sin(grid),
                           0.77 + 0.02 * np.sin(grid))},
        "DeepHit": {"times": grid, "values": 0.79 + 0.01 * np.cos(grid)},
    }


def test_single_curve_render_returns_display(grid):
    disp = ts.plot.auc_over_time(grid, 0.75 + 0.02 * np.sin(grid))
    assert disp.fig is not None
    assert disp.ax is not None
    assert list(disp.lines.keys()) == [""]
    assert disp.ci_polys == {}


def test_single_curve_with_ci_creates_band(grid):
    v = 0.75 + 0.02 * np.sin(grid)
    disp = ts.plot.auc_over_time(grid, v, ci=(v - 0.03, v + 0.03), label="Cox")
    assert list(disp.ci_polys.keys()) == ["Cox"]
    verts = disp.ci_polys["Cox"].get_paths()[0].vertices
    # Band envelope lies inside the expected ±0.03 corridor.
    assert verts[:, 1].min() >= v.min() - 0.05
    assert verts[:, 1].max() <= v.max() + 0.05


def test_multi_model_overlay_renders_one_line_per_model(models):
    disp = ts.plot.auc_over_time(models=models)
    assert list(disp.lines.keys()) == ["Cox", "DeepHit"]
    assert list(disp.ci_polys.keys()) == ["Cox"]


def test_reference_line_drawn_for_auc_by_default(grid):
    disp = ts.plot.auc_over_time(grid, 0.7 + 0 * grid)
    assert disp.reference is not None
    assert disp.reference.get_ydata()[0] == pytest.approx(0.5)


def test_reference_line_suppressed_when_none(grid):
    disp = ts.plot.auc_over_time(grid, 0.7 + 0 * grid, reference=None)
    assert disp.reference is None


def test_brier_has_no_reference_by_default(grid):
    disp = ts.plot.brier_over_time(grid, 0.1 + 0 * grid)
    assert disp.reference is None


def test_concordance_has_reference_at_half(grid):
    disp = ts.plot.concordance_over_time(grid, 0.7 + 0 * grid)
    assert disp.reference is not None
    assert disp.reference.get_ydata()[0] == pytest.approx(0.5)


def test_ylim_applied(grid):
    disp = ts.plot.brier_over_time(grid, 0.1 + 0 * grid)
    lo, hi = disp.ax.get_ylim()
    assert (lo, hi) == pytest.approx((0.0, 0.25))


def test_user_ax_is_respected(grid):
    _, ax = plt.subplots()
    disp = ts.plot.auc_over_time(grid, 0.7 + 0 * grid, ax=ax)
    assert disp.ax is ax


def test_models_and_single_curve_are_mutually_exclusive(grid, models):
    with pytest.raises(ValueError, match="single-curve"):
        ts.plot.auc_over_time(grid, 0.7 + 0 * grid, models=models)


def test_color_label_invalid_with_models(models):
    with pytest.raises(ValueError, match="single-curve only"):
        ts.plot.auc_over_time(models=models, color="red")


def test_missing_inputs_raise():
    with pytest.raises(ValueError, match="must be supplied"):
        ts.plot.auc_over_time()


# ----- folds= input -----

def test_folds_single_curve_renders_mean_and_band(grid):
    rng = np.random.default_rng(0)
    fold_matrix = 0.78 + 0.02 * np.sin(grid) + 0.03 * rng.normal(size=(5, grid.size))
    disp = ts.plot.auc_over_time(grid, folds=fold_matrix, label="Cox")
    assert list(disp.lines.keys()) == ["Cox"]
    assert list(disp.ci_polys.keys()) == ["Cox"]
    # Rendered line = fold mean
    np.testing.assert_allclose(
        disp.lines["Cox"].get_ydata(), fold_matrix.mean(axis=0), atol=1e-12,
    )


def test_folds_default_band_is_sd(grid):
    rng = np.random.default_rng(0)
    fold_matrix = 0.75 + 0.05 * rng.normal(size=(6, grid.size))
    disp = ts.plot.auc_over_time(grid, folds=fold_matrix)
    verts = disp.ci_polys[""].get_paths()[0].vertices
    mean = fold_matrix.mean(axis=0)
    std = fold_matrix.std(axis=0, ddof=1)
    # The polygon path includes lower edge + reversed upper edge; the y
    # range of the polygon should encompass mean - std to mean + std.
    assert verts[:, 1].min() == pytest.approx((mean - std).min(), abs=1e-8)
    assert verts[:, 1].max() == pytest.approx((mean + std).max(), abs=1e-8)


def test_folds_band_se_is_narrower_than_sd(grid):
    rng = np.random.default_rng(1)
    fold_matrix = 0.75 + 0.05 * rng.normal(size=(10, grid.size))
    sd = ts.plot.auc_over_time(grid, folds=fold_matrix, band="sd")
    se = ts.plot.auc_over_time(grid, folds=fold_matrix, band="se")
    sd_verts = sd.ci_polys[""].get_paths()[0].vertices
    se_verts = se.ci_polys[""].get_paths()[0].vertices
    sd_span = sd_verts[:, 1].max() - sd_verts[:, 1].min()
    se_span = se_verts[:, 1].max() - se_verts[:, 1].min()
    assert se_span < sd_span


def test_folds_models_overlay(grid):
    rng = np.random.default_rng(2)
    f_cox = 0.75 + 0.03 * rng.normal(size=(5, grid.size))
    disp = ts.plot.auc_over_time(models={
        "Cox":     {"times": grid, "folds": f_cox},
        "DeepHit": {"times": grid, "values": 0.80 + 0 * grid},
    })
    assert list(disp.lines.keys()) == ["Cox", "DeepHit"]
    # Only Cox has a band -- DeepHit was values-only.
    assert list(disp.ci_polys.keys()) == ["Cox"]


def test_folds_unknown_band_raises(grid):
    fold_matrix = np.zeros((3, grid.size))
    with pytest.raises(ValueError, match="unknown band"):
        ts.plot.auc_over_time(grid, folds=fold_matrix, band="iqr")


def test_folds_too_few_rows_raises(grid):
    fold_matrix = np.zeros((1, grid.size))
    with pytest.raises(ValueError, match="at least 2"):
        ts.plot.auc_over_time(grid, folds=fold_matrix)


def test_folds_wrong_ncols_raises(grid):
    fold_matrix = np.zeros((4, grid.size + 1))
    with pytest.raises(ValueError, match=r"\(n_folds, n_times\)"):
        ts.plot.auc_over_time(grid, folds=fold_matrix)


def test_folds_mutually_exclusive_with_values(grid):
    fold_matrix = np.zeros((4, grid.size))
    with pytest.raises(ValueError, match="mutually exclusive"):
        ts.plot.auc_over_time(grid, 0.7 + 0 * grid, folds=fold_matrix)


def test_folds_mutually_exclusive_with_ci(grid):
    fold_matrix = np.zeros((4, grid.size))
    v = 0.7 + 0 * grid
    with pytest.raises(ValueError, match="mutually exclusive"):
        ts.plot.auc_over_time(
            grid, folds=fold_matrix, ci=(v - 0.01, v + 0.01),
        )


def test_folds_works_for_brier_and_concordance(grid):
    rng = np.random.default_rng(3)
    fold_matrix = 0.12 + 0.02 * rng.normal(size=(5, grid.size))
    disp_brier = ts.plot.brier_over_time(grid, folds=fold_matrix)
    assert list(disp_brier.ci_polys.keys()) == [""]

    fold_matrix_c = 0.72 + 0.02 * rng.normal(size=(5, grid.size))
    disp_c = ts.plot.concordance_over_time(grid, folds=fold_matrix_c)
    assert list(disp_c.ci_polys.keys()) == [""]


def test_times_values_shape_mismatch_raises(grid):
    with pytest.raises(ValueError, match="equal length"):
        ts.plot.auc_over_time(grid, np.array([0.7, 0.8]))


def test_ci_shape_mismatch_raises(grid):
    v = 0.7 + 0 * grid
    with pytest.raises(ValueError, match="bounds must match"):
        ts.plot.auc_over_time(grid, v, ci=(np.array([0.6]), np.array([0.8])))


def test_legend_only_with_multiple_named_curves(grid, models):
    disp_single = ts.plot.auc_over_time(grid, 0.7 + 0 * grid, label="Cox")
    assert disp_single.ax.get_legend() is None
    disp_multi = ts.plot.auc_over_time(models=models)
    assert disp_multi.ax.get_legend() is not None
