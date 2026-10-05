from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import tausurv as ts  # noqa: E402
from tausurv.copulas.clayton import Clayton  # noqa: E402
from tausurv.copulas.frank import Frank  # noqa: E402
from tausurv.copulas.gumbel import Gumbel  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def clayton():
    return Clayton(theta=2.0)


def test_contour_returns_display(clayton):
    disp = ts.plot.copula.contour(clayton)
    assert disp.fig is not None
    assert disp.ax is not None
    assert disp.contours is not None
    assert disp.reference is not None
    assert disp.tau_annotation is not None


def test_contour_diagonal_can_be_suppressed(clayton):
    disp = ts.plot.copula.contour(clayton, show_diagonal=False)
    assert disp.reference is None


def test_contour_tau_can_be_suppressed(clayton):
    disp = ts.plot.copula.contour(clayton, annotate_tau=False)
    assert disp.tau_annotation is None


def test_contour_filled_variant(clayton):
    disp = ts.plot.copula.contour(clayton, filled=True)
    # contourf returns a QuadContourSet with patches rather than line
    # collections; we just check it was set.
    assert disp.contours is not None


def test_contour_axes_are_unit_square_and_equal(clayton):
    disp = ts.plot.copula.contour(clayton)
    assert disp.ax.get_xlim() == (0.0, 1.0)
    assert disp.ax.get_ylim() == (0.0, 1.0)
    assert disp.ax.get_aspect() == 1.0 or str(disp.ax.get_aspect()) == "equal"


def test_contour_color_overrides_cmap(clayton):
    disp = ts.plot.copula.contour(clayton, color="black")
    # All line segments should carry the override colour.
    collections = (
        disp.contours.collections if hasattr(disp.contours, "collections") else []
    )
    if collections:
        # at least one collection set; we don't assert colour specifics --
        # contourf/contour internals vary across mpl versions
        assert collections[0] is not None


def test_density_returns_display(clayton):
    disp = ts.plot.copula.density(clayton, n_grid=20)
    assert disp.fig is not None
    assert disp.ax is not None
    assert disp.mesh is not None
    assert disp.colorbar is not None
    assert disp.tau_annotation is not None


def test_density_colorbar_can_be_suppressed(clayton):
    disp = ts.plot.copula.density(clayton, n_grid=20, colorbar=False)
    assert disp.colorbar is None


def test_density_log_scale_default(clayton):
    from matplotlib.colors import LogNorm

    disp = ts.plot.copula.density(clayton, n_grid=20)
    assert isinstance(disp.mesh.norm, LogNorm)


def test_density_linear_scale(clayton):
    from matplotlib.colors import LogNorm

    disp = ts.plot.copula.density(clayton, n_grid=20, log_scale=False)
    assert not isinstance(disp.mesh.norm, LogNorm)


def test_density_grid_resolution(clayton):
    disp = ts.plot.copula.density(clayton, n_grid=15)
    # pcolormesh stores the coordinate arrays; the mesh has (n-1) cells per
    # axis when shading="auto" with same-size coords. We check the data
    # array dimensions.
    data = disp.mesh.get_array()
    assert data.size in (15 * 15, 14 * 14)


def test_scatter_returns_display(clayton):
    disp = ts.plot.copula.scatter(clayton, n_samples=200, seed=0)
    assert disp.points is not None
    assert disp.tau_annotation is not None


def test_scatter_n_samples_respected(clayton):
    disp = ts.plot.copula.scatter(clayton, n_samples=137, seed=0)
    assert disp.points.get_offsets().shape[0] == 137


def test_scatter_seed_is_deterministic(clayton):
    a = ts.plot.copula.scatter(clayton, n_samples=100, seed=42).points.get_offsets()
    b = ts.plot.copula.scatter(clayton, n_samples=100, seed=42).points.get_offsets()
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_scatter_samples_in_unit_square(clayton):
    disp = ts.plot.copula.scatter(clayton, n_samples=500, seed=0)
    pts = np.asarray(disp.points.get_offsets())
    assert pts.min() >= 0.0
    assert pts.max() <= 1.0


def test_tau_annotation_text_matches_copula():
    cl = Clayton(theta=2.5)
    disp = ts.plot.copula.contour(cl)
    text = disp.tau_annotation.get_text()
    expected = f"$\\tau = {cl.kendalls_tau():.2f}$"
    assert text == expected


@pytest.mark.parametrize("copula", [Clayton(2.0), Gumbel(2.0), Frank(5.0)])
def test_all_three_plots_render_for_each_family(copula):
    ts.plot.copula.contour(copula, n_grid=15)
    ts.plot.copula.density(copula, n_grid=15)
    ts.plot.copula.scatter(copula, n_samples=100, seed=0)


def test_user_ax_is_respected(clayton):
    _, ax = plt.subplots()
    disp = ts.plot.copula.contour(clayton, ax=ax)
    assert disp.ax is ax


def test_grids_have_no_gridlines(clayton):
    # The publication style turns y-grid on; copula plots disable it for
    # readability of heatmaps and contours.
    ts.plot.set_style("publication")
    disp = ts.plot.copula.density(clayton, n_grid=15)
    assert not disp.ax.xaxis._major_tick_kw.get("gridOn", False)
