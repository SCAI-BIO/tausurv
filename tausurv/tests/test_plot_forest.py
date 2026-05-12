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
def simple_hrs():
    names = ["A", "B", "C", "D"]
    hr = np.array([1.2, 0.7, 1.8, 1.0])
    lo = hr * 0.8
    hi = hr * 1.25
    return names, hr, lo, hi


def test_returns_forest_display(simple_hrs):
    names, hr, lo, hi = simple_hrs
    disp = ts.plot.forest(names, hr, ci=(lo, hi))
    assert disp.fig is not None
    assert disp.ax is not None
    assert disp.points is not None
    assert disp.ci_lines is not None
    assert disp.reference is not None


def test_no_ci_yields_no_ci_lines(simple_hrs):
    names, hr, _, _ = simple_hrs
    disp = ts.plot.forest(names, hr)
    assert disp.ci_lines is None


def test_reference_defaults_to_one(simple_hrs):
    names, hr, lo, hi = simple_hrs
    disp = ts.plot.forest(names, hr, ci=(lo, hi))
    assert disp.reference.get_xdata()[0] == pytest.approx(1.0)


def test_reference_can_be_suppressed(simple_hrs):
    names, hr, _, _ = simple_hrs
    disp = ts.plot.forest(names, hr, reference=None)
    assert disp.reference is None


def test_reference_custom_value(simple_hrs):
    names, hr, _, _ = simple_hrs
    disp = ts.plot.forest(names, hr, reference=2.0)
    assert disp.reference.get_xdata()[0] == pytest.approx(2.0)


def test_annotations_match_row_count(simple_hrs):
    names, hr, lo, hi = simple_hrs
    disp = ts.plot.forest(names, hr, ci=(lo, hi))
    assert len(disp.annotations) == len(names)


def test_annotate_false_yields_no_annotations(simple_hrs):
    names, hr, lo, hi = simple_hrs
    disp = ts.plot.forest(names, hr, ci=(lo, hi), annotate=False)
    assert disp.annotations == []


def test_y_axis_inverted_first_variable_on_top(simple_hrs):
    names, hr, _, _ = simple_hrs
    disp = ts.plot.forest(names, hr)
    # Inverted means y-lim is reversed: ylim[0] > ylim[1].
    lo, hi = disp.ax.get_ylim()
    assert lo > hi


def test_y_tick_labels_match_names(simple_hrs):
    names, hr, _, _ = simple_hrs
    disp = ts.plot.forest(names, hr)
    labels = [t.get_text() for t in disp.ax.get_yticklabels()]
    assert labels == list(names)


def test_sort_input_preserves_order(simple_hrs):
    names, hr, _, _ = simple_hrs
    disp = ts.plot.forest(names, hr, sort="input")
    labels = [t.get_text() for t in disp.ax.get_yticklabels()]
    assert labels == list(names)


def test_sort_estimate_ascending(simple_hrs):
    names, hr, _, _ = simple_hrs
    disp = ts.plot.forest(names, hr, sort="estimate")
    labels = [t.get_text() for t in disp.ax.get_yticklabels()]
    expected = list(np.asarray(names)[np.argsort(hr)])
    assert labels == expected


def test_sort_name_alphabetical():
    names = ["Gamma", "Alpha", "Beta"]
    hr = np.array([1.0, 2.0, 3.0])
    disp = ts.plot.forest(names, hr, sort="name")
    labels = [t.get_text() for t in disp.ax.get_yticklabels()]
    assert labels == ["Alpha", "Beta", "Gamma"]


def test_unknown_sort_raises(simple_hrs):
    names, hr, _, _ = simple_hrs
    with pytest.raises(ValueError, match="unknown sort"):
        ts.plot.forest(names, hr, sort="bogus")


def test_log_scale_default_is_log(simple_hrs):
    names, hr, _, _ = simple_hrs
    disp = ts.plot.forest(names, hr)
    assert disp.ax.get_xscale() == "log"


def test_linear_scale_for_risk_differences():
    names = ["A", "B"]
    rd = np.array([-0.05, 0.10])
    disp = ts.plot.forest(
        names, rd, ci=(rd - 0.03, rd + 0.03),
        log_scale=False, reference=0.0, xlabel="Risk difference",
    )
    assert disp.ax.get_xscale() == "linear"
    assert disp.reference.get_xdata()[0] == 0.0


def test_log_scale_rejects_non_positive_estimates(simple_hrs):
    names, hr, _, _ = simple_hrs
    hr[1] = -0.5
    with pytest.raises(ValueError, match="positive"):
        ts.plot.forest(names, hr)


def test_log_scale_rejects_non_positive_ci_bounds():
    names = ["A", "B"]
    hr = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        ts.plot.forest(
            names, hr, ci=(np.array([0.0, 1.5]), np.array([1.5, 2.5])),
        )


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        ts.plot.forest(["A", "B"], np.array([1.0]))


def test_ci_shape_mismatch_raises():
    with pytest.raises(ValueError, match="must match"):
        ts.plot.forest(
            ["A", "B"], np.array([1.0, 2.0]),
            ci=(np.array([0.5]), np.array([1.5])),
        )


def test_annotation_format_applied(simple_hrs):
    names, hr, lo, hi = simple_hrs
    disp = ts.plot.forest(
        names, hr, ci=(lo, hi),
        annotation_format="{:.1f} [{:.1f}-{:.1f}]",
    )
    first = disp.annotations[0].get_text()
    assert first == f"{hr[0]:.1f} [{lo[0]:.1f}-{hi[0]:.1f}]"


def test_user_ax_is_respected(simple_hrs):
    names, hr, _, _ = simple_hrs
    _, ax = plt.subplots()
    disp = ts.plot.forest(names, hr, ax=ax)
    assert disp.ax is ax
