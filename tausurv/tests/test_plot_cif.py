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
def cr_data():
    # 6 subjects, 2 causes, one censoring.
    Y = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 5.0])
    E = np.array([1, 2, 1, 0, 2, 1], dtype=np.int64)
    return Y, E


def test_returns_cif_display(cr_data):
    Y, E = cr_data
    disp = ts.plot.cif(Y, E)
    assert disp.fig is not None
    assert disp.ax is not None
    assert disp.at_risk_ax is not None
    assert set(disp.lines.keys()) == {"Cause 1", "Cause 2"}


def test_causes_auto_inferred_in_ascending_order(cr_data):
    Y, E = cr_data
    disp = ts.plot.cif(Y, E)
    assert list(disp.lines.keys()) == ["Cause 1", "Cause 2"]


def test_explicit_single_cause_yields_unlabelled_curve(cr_data):
    Y, E = cr_data
    disp = ts.plot.cif(Y, E, causes=1)
    assert list(disp.lines.keys()) == [""]


def test_explicit_causes_list(cr_data):
    Y, E = cr_data
    disp = ts.plot.cif(Y, E, causes=[2])
    assert list(disp.lines.keys()) == [""]


def test_cause_labels_applied(cr_data):
    Y, E = cr_data
    disp = ts.plot.cif(Y, E, cause_labels={1: "Relapse", 2: "Death"})
    assert list(disp.lines.keys()) == ["Relapse", "Death"]


def test_cif_values_match_aalen_johansen(cr_data):
    Y, E = cr_data
    disp = ts.plot.cif(Y, E, causes=1, at_risk=False)
    line = disp.lines[""]
    aj = ts.aalen_johansen(Y, E, 1)
    # The rendered line includes a prepended (t=0, F=0) anchor before AJ.
    np.testing.assert_array_equal(line.get_xdata()[1:], aj.time)
    np.testing.assert_allclose(line.get_ydata()[1:], aj.value)
    assert line.get_xdata()[0] == 0.0
    assert line.get_ydata()[0] == 0.0


def test_at_risk_disabled_yields_no_table_axes(cr_data):
    Y, E = cr_data
    disp = ts.plot.cif(Y, E, at_risk=False)
    assert disp.at_risk_ax is None


def test_user_supplied_ax_disables_at_risk(cr_data):
    Y, E = cr_data
    _, ax = plt.subplots()
    disp = ts.plot.cif(Y, E, ax=ax)
    assert disp.ax is ax
    assert disp.at_risk_ax is None


def test_group_with_single_cause_produces_per_group_rows(cr_data):
    Y, E = cr_data
    G = np.array(["a", "a", "a", "b", "b", "b"])
    disp = ts.plot.cif(Y, E, causes=1, group=G)
    assert list(disp.lines.keys()) == ["a", "b"]


def test_group_with_multi_cause_raises(cr_data):
    Y, E = cr_data
    G = np.array(["a"] * 6)
    with pytest.raises(ValueError, match="single cause"):
        ts.plot.cif(Y, E, causes=[1, 2], group=G)


def test_default_causes_with_group_uses_first_observed(cr_data):
    # When group= is set and causes is None and there are multiple observed
    # causes, the rule (group is only valid with one cause) bites.
    Y, E = cr_data
    G = np.array(["a"] * 6)
    with pytest.raises(ValueError, match="single cause"):
        ts.plot.cif(Y, E, group=G)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        ts.plot.cif([1.0, 2.0, 3.0], [1, 2])


def test_group_shape_mismatch_raises(cr_data):
    Y, E = cr_data
    with pytest.raises(ValueError, match="group must match"):
        ts.plot.cif(Y, E, causes=1, group=["a", "b"])


def test_no_events_raises():
    Y = np.array([1.0, 2.0, 3.0])
    E = np.array([0, 0, 0], dtype=np.int64)
    with pytest.raises(ValueError, match="no events"):
        ts.plot.cif(Y, E)


def test_non_positive_cause_raises(cr_data):
    Y, E = cr_data
    with pytest.raises(ValueError, match="positive integer"):
        ts.plot.cif(Y, E, causes=[1, 0])


def test_empty_causes_list_raises(cr_data):
    Y, E = cr_data
    with pytest.raises(ValueError, match="non-empty"):
        ts.plot.cif(Y, E, causes=[])


def test_censor_ticks_off_produces_no_marker(cr_data):
    Y, E = cr_data
    disp = ts.plot.cif(Y, E, causes=1, censor_ticks=False, at_risk=False)
    marker_lines = [
        ln for ln in disp.ax.get_lines() if ln.get_marker() == "|"
    ]
    assert marker_lines == []


def test_censor_ticks_on_renders_marker(cr_data):
    Y, E = cr_data
    disp = ts.plot.cif(Y, E, causes=1, censor_ticks=True, at_risk=False)
    marker_lines = [
        ln for ln in disp.ax.get_lines() if ln.get_marker() == "|"
    ]
    # One censoring in the fixture -> one marker line, with one tick.
    assert len(marker_lines) == 1
    assert marker_lines[0].get_xdata().size == 1


def test_legend_only_with_multiple_named_curves(cr_data):
    Y, E = cr_data
    # Multi-cause: 2 lines, legend should appear.
    disp_multi = ts.plot.cif(Y, E)
    assert disp_multi.ax.get_legend() is not None
    # Single cause, no group: 1 unlabelled line, no legend.
    disp_single = ts.plot.cif(Y, E, causes=1)
    assert disp_single.ax.get_legend() is None


def test_ylim_clamped_to_unit_interval(cr_data):
    Y, E = cr_data
    disp = ts.plot.cif(Y, E)
    lo, hi = disp.ax.get_ylim()
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(1.02)
