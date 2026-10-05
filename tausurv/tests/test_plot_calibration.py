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
def well_calibrated():
    predicted = np.linspace(0.05, 0.95, 10)
    observed = predicted + 0.01 * np.sin(predicted * 6)
    return predicted, observed


def test_returns_calibration_display(well_calibrated):
    p, o = well_calibrated
    disp = ts.plot.calibration(p, o, label="Cox")
    assert disp.fig is not None
    assert disp.ax is not None
    assert list(disp.lines.keys()) == ["Cox"]
    assert disp.reference is not None


def test_reference_diagonal_spans_axis_upper(well_calibrated):
    p, o = well_calibrated
    disp = ts.plot.calibration(p, o)
    xs = disp.reference.get_xdata()
    ys = disp.reference.get_ydata()
    np.testing.assert_array_equal(xs, ys)
    assert xs[0] == 0.0
    assert xs[1] > max(p.max(), o.max())


def test_reference_can_be_suppressed(well_calibrated):
    p, o = well_calibrated
    disp = ts.plot.calibration(p, o, reference=False)
    assert disp.reference is None


def test_axes_are_square_by_default(well_calibrated):
    p, o = well_calibrated
    disp = ts.plot.calibration(p, o)
    assert disp.ax.get_aspect() == 1.0 or str(disp.ax.get_aspect()) == "equal"


def test_aspect_equal_can_be_disabled(well_calibrated):
    p, o = well_calibrated
    disp = ts.plot.calibration(p, o, aspect_equal=False)
    assert str(disp.ax.get_aspect()) != "equal"


def test_axis_upper_has_padding_and_floor():
    # Very small data -> floor at 0.05.
    p = np.array([0.001, 0.002, 0.003])
    disp = ts.plot.calibration(p, p)
    xlim_hi = disp.ax.get_xlim()[1]
    assert xlim_hi == pytest.approx(0.05)


def test_axis_upper_pads_above_data():
    p = np.array([0.1, 0.2, 0.4])
    disp = ts.plot.calibration(p, p)
    assert disp.ax.get_xlim()[1] == pytest.approx(0.4 * 1.15)


def test_multi_model_overlay():
    p1 = np.linspace(0.1, 0.8, 8)
    p2 = np.linspace(0.05, 0.9, 8)
    models = {
        "Cox": {"predicted": p1, "observed": p1 + 0.02},
        "DeepHit": {"predicted": p2, "observed": p2 - 0.03},
    }
    disp = ts.plot.calibration(models=models)
    assert list(disp.lines.keys()) == ["Cox", "DeepHit"]
    # Combined upper bound should reflect the larger of the two models' maxima.
    assert disp.ax.get_xlim()[1] == pytest.approx(0.9 * 1.15)


def test_points_sorted_by_predicted(well_calibrated):
    # Shuffle the order; the rendered line should still be drawn from low to
    # high predicted so the trajectory is monotone-x.
    p, o = well_calibrated
    perm = np.array([4, 1, 7, 0, 9, 3, 6, 2, 8, 5])
    disp = ts.plot.calibration(p[perm], o[perm], label="A")
    line = disp.lines["A"]
    xdata = line.get_xdata()
    assert np.all(np.diff(xdata) >= 0)


def test_models_and_single_curve_mutually_exclusive():
    p = np.linspace(0.1, 0.9, 5)
    with pytest.raises(ValueError, match="not both"):
        ts.plot.calibration(p, p, models={"X": {"predicted": p, "observed": p}})


def test_color_label_invalid_with_models():
    p = np.linspace(0.1, 0.9, 5)
    with pytest.raises(ValueError, match="single-curve only"):
        ts.plot.calibration(
            models={"X": {"predicted": p, "observed": p}},
            color="red",
        )


def test_missing_inputs_raise():
    with pytest.raises(ValueError, match="must be supplied"):
        ts.plot.calibration()


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        ts.plot.calibration([0.1, 0.2, 0.3], [0.1, 0.2])


def test_legend_only_with_multiple_named_curves(well_calibrated):
    p, o = well_calibrated
    disp_single = ts.plot.calibration(p, o, label="Cox")
    assert disp_single.ax.get_legend() is None
    disp_multi = ts.plot.calibration(
        models={
            "A": {"predicted": p, "observed": o},
            "B": {"predicted": p, "observed": o * 0.9},
        },
    )
    assert disp_multi.ax.get_legend() is not None


def test_user_ax_is_respected(well_calibrated):
    p, o = well_calibrated
    _, ax = plt.subplots()
    disp = ts.plot.calibration(p, o, ax=ax)
    assert disp.ax is ax
