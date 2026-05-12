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
    Y = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    E = np.array([1, 2, 1, 0, 2, 1, 2, 0], dtype=np.int64)
    return Y, E


def test_returns_display_with_bands_per_cause_plus_survival(cr_data):
    Y, E = cr_data
    disp = ts.plot.stacked_cif(Y, E)
    assert disp.fig is not None
    assert disp.ax is not None
    # Two observed causes + survival.
    assert set(disp.bands.keys()) == {"Cause 1", "Cause 2", "Survival"}


def test_cause_labels_applied(cr_data):
    Y, E = cr_data
    disp = ts.plot.stacked_cif(
        Y, E, cause_labels={1: "Relapse", 2: "Death"},
    )
    assert "Relapse" in disp.bands
    assert "Death" in disp.bands


def test_show_survival_false_omits_band(cr_data):
    Y, E = cr_data
    disp = ts.plot.stacked_cif(Y, E, show_survival=False)
    assert "Survival" not in disp.bands


def test_explicit_causes_list(cr_data):
    Y, E = cr_data
    disp = ts.plot.stacked_cif(Y, E, causes=[2])
    assert set(disp.bands.keys()) == {"Cause 2", "Survival"}


def test_explicit_single_cause(cr_data):
    Y, E = cr_data
    disp = ts.plot.stacked_cif(Y, E, causes=1)
    assert "Cause 1" in disp.bands
    assert "Cause 2" not in disp.bands


def test_y_axis_unit_interval(cr_data):
    Y, E = cr_data
    disp = ts.plot.stacked_cif(Y, E)
    assert disp.ax.get_ylim() == (0.0, 1.0)


def test_stack_sums_to_one_at_every_observation_time(cr_data):
    Y, E = cr_data
    disp = ts.plot.stacked_cif(Y, E)
    # The top boundary of the Survival band's path should hit 1.0 (modulo
    # float noise) at every observed time. Read off via the band's vertices.
    # Easier: sample the cumulative incidences and the KM survival at the
    # observation times and verify they sum to 1.
    aj1 = ts.aalen_johansen(Y, E, 1)(np.concatenate([[0.0], np.unique(Y)]))
    aj2 = ts.aalen_johansen(Y, E, 2)(np.concatenate([[0.0], np.unique(Y)]))
    km = ts.kaplan_meier(Y, np.clip(E, 0, 1))(np.concatenate([[0.0], np.unique(Y)]))
    total = aj1 + aj2 + km
    np.testing.assert_allclose(total, 1.0, atol=1e-9)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        ts.plot.stacked_cif([1.0, 2.0, 3.0], [1, 2])


def test_no_events_raises():
    with pytest.raises(ValueError, match="no events"):
        ts.plot.stacked_cif(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0]))


def test_non_positive_cause_raises(cr_data):
    Y, E = cr_data
    with pytest.raises(ValueError, match="positive integer"):
        ts.plot.stacked_cif(Y, E, causes=[0])


def test_empty_causes_list_raises(cr_data):
    Y, E = cr_data
    with pytest.raises(ValueError, match="non-empty"):
        ts.plot.stacked_cif(Y, E, causes=[])


def test_user_ax_is_respected(cr_data):
    Y, E = cr_data
    _, ax = plt.subplots()
    disp = ts.plot.stacked_cif(Y, E, ax=ax)
    assert disp.ax is ax
