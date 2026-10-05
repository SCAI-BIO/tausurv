from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import tausurv as ts  # noqa: E402
from tausurv.step import StepFunction  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_km_returns_display_with_curve_axes():
    disp = ts.plot.km([1.0, 2.0, 3.0], [1, 1, 1])
    assert disp.fig is not None
    assert disp.ax is not None
    assert disp.at_risk_ax is not None
    assert list(disp.lines.keys()) == [""]


def test_km_survival_values_match_kaplan_meier():
    Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    D = np.array([1, 0, 1, 1, 0], dtype=np.int8)
    disp = ts.plot.km(Y, D, ci=False, at_risk=False)
    line = disp.lines[""]

    # The rendered line is the prepended (t=0, S=1) anchor + the KM step grid.
    km = ts.kaplan_meier(Y, D)
    np.testing.assert_array_equal(line.get_xdata()[1:], km.time)
    np.testing.assert_allclose(line.get_ydata()[1:], km.value)
    assert line.get_xdata()[0] == 0.0
    assert line.get_ydata()[0] == 1.0


def test_km_logit_ci_inside_unit_interval():
    rng = np.random.default_rng(0)
    Y = rng.exponential(2.0, 50)
    D = (rng.uniform(size=50) < 0.7).astype(np.int8)
    disp = ts.plot.km(Y, D, ci_method="logit", at_risk=False)
    verts = disp.ci_polys[""].get_paths()[0].vertices
    assert verts[:, 1].min() >= 0.0
    assert verts[:, 1].max() <= 1.0


@pytest.mark.parametrize("method", ["logit", "log-log", "wald"])
def test_km_ci_methods_render_inside_unit_interval(method):
    rng = np.random.default_rng(1)
    Y = rng.exponential(2.0, 40)
    D = (rng.uniform(size=40) < 0.6).astype(np.int8)
    disp = ts.plot.km(Y, D, ci_method=method, at_risk=False)
    verts = disp.ci_polys[""].get_paths()[0].vertices
    assert verts[:, 1].min() >= -1e-9
    assert verts[:, 1].max() <= 1.0 + 1e-9


def test_km_group_produces_one_line_per_group_in_first_seen_order():
    Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    D = np.array([1, 1, 1, 1, 1, 1], dtype=np.int8)
    G = np.array(["b", "a", "b", "a", "b", "a"])
    disp = ts.plot.km(Y, D, group=G, at_risk=False)
    assert list(disp.lines.keys()) == ["b", "a"]
    assert list(disp.ci_polys.keys()) == ["b", "a"]


def test_km_at_risk_disabled_yields_no_table_axes():
    disp = ts.plot.km([1.0, 2.0, 3.0], [1, 1, 1], at_risk=False)
    assert disp.at_risk_ax is None


def test_km_user_supplied_ax_disables_at_risk_table():
    _, ax = plt.subplots()
    disp = ts.plot.km([1.0, 2.0, 3.0], [1, 1, 1], ax=ax)
    assert disp.ax is ax
    assert disp.at_risk_ax is None


def test_km_no_ci_polys_when_ci_false():
    disp = ts.plot.km([1.0, 2.0, 3.0], [1, 1, 1], ci=False, at_risk=False)
    assert disp.ci_polys == {}


def test_km_censor_ticks_off_produces_no_marker_collection():
    # n=200 -> default censor_ticks is auto-off. Verify explicit False stays off.
    rng = np.random.default_rng(2)
    Y = rng.exponential(2.0, 30)
    D = np.zeros(30, dtype=np.int8)
    D[:15] = 1
    disp = ts.plot.km(Y, D, censor_ticks=False, ci=False, at_risk=False)
    # The line itself uses linestyle="-"; censor ticks would be a second
    # Line2D with marker="|" and linestyle="None". With censor_ticks=False
    # only the step line should exist on the axes.
    marker_lines = [ln for ln in disp.ax.get_lines() if ln.get_marker() == "|"]
    assert marker_lines == []


def test_km_censor_ticks_on_produces_marker_line():
    rng = np.random.default_rng(3)
    Y = rng.exponential(2.0, 30)
    D = np.zeros(30, dtype=np.int8)
    D[:15] = 1
    disp = ts.plot.km(Y, D, censor_ticks=True, ci=False, at_risk=False)
    marker_lines = [ln for ln in disp.ax.get_lines() if ln.get_marker() == "|"]
    assert len(marker_lines) == 1
    assert marker_lines[0].get_xdata().size == (D == 0).sum()


def test_km_stepfunction_path_returns_curve_only():
    Y = np.array([1.0, 2.0, 3.0])
    D = np.array([1, 1, 1], dtype=np.int8)
    S = ts.kaplan_meier(Y, D)
    disp = ts.plot.km(S, label="A")
    assert isinstance(S, StepFunction)
    assert disp.at_risk_ax is None
    assert disp.ci_polys == {}
    assert list(disp.lines.keys()) == ["A"]


def test_km_stepfunction_rejects_event_indicator():
    S = ts.kaplan_meier([1.0, 2.0], [1, 1])
    with pytest.raises(ValueError, match="StepFunction"):
        ts.plot.km(S, [1, 0])


def test_km_stepfunction_rejects_group():
    S = ts.kaplan_meier([1.0, 2.0], [1, 1])
    with pytest.raises(ValueError, match="StepFunction"):
        ts.plot.km(S, group=["a", "b"])


def test_km_raw_arrays_without_event_indicator_raises():
    with pytest.raises(ValueError, match="event_indicator"):
        ts.plot.km([1.0, 2.0])


def test_km_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        ts.plot.km([1.0, 2.0, 3.0], [1, 0])


def test_km_group_shape_mismatch_raises():
    with pytest.raises(ValueError, match="group must match"):
        ts.plot.km([1.0, 2.0, 3.0], [1, 1, 1], group=["a", "b"])


def test_km_group_with_color_raises():
    with pytest.raises(ValueError, match="single-curve"):
        ts.plot.km(
            [1.0, 2.0],
            [1, 1],
            group=["a", "b"],
            color="red",
        )


def test_km_at_risk_table_reports_known_counts():
    # 5 subjects: events at t=1, 3, 5; censored at t=2, 4.
    # At t=0: 5 at risk; at t=1: 5; at t=2: 4; at t=3: 3; at t=4: 2; at t=5: 1.
    Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    D = np.array([1, 0, 1, 0, 1], dtype=np.int8)
    disp = ts.plot.km(Y, D, at_risk=True, ci=False)
    table_ax = disp.at_risk_ax
    assert table_ax is not None
    # The rendered ints under each tick are text artists; extract them.
    ints = {t.get_text() for t in table_ax.texts if t.get_text().isdigit()}
    # At t=0 we have 5 at risk; at t=5 we have 1. Both should appear.
    assert "5" in ints
    assert "1" in ints
