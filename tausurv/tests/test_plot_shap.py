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
def shap_3d():
    rng = np.random.default_rng(0)
    n_subjects, n_features, n_times = 30, 5, 20
    times = np.linspace(0.5, 5.0, n_times)
    features = ["A", "B", "C", "D", "E"]
    # B and C dominate; A weakly positive; D, E noise
    template = np.stack(
        [
            0.001 + 0 * times,
            0.04 * np.exp(-((times - 2.0) ** 2)),
            -0.05 * np.exp(-((times - 3.0) ** 2)),
            0.005 * np.sin(times),
            0.003 * np.cos(times),
        ]
    )
    values = template[None] + 0.002 * rng.normal(size=(n_subjects, n_features, n_times))
    baseline = np.exp(-0.15 * times)
    return values, baseline, times, features


def test_curves_returns_display(shap_3d):
    values, _, times, features = shap_3d
    disp = ts.plot.shap.curves(values, times, features, subject=0, top_k=3)
    assert disp.fig is not None
    assert disp.ax is not None
    assert disp.reference is not None
    assert len(disp.lines) == 3
    # 5 features - top_k=3 = 2 faded
    assert len(disp.faded_lines) == 2


def test_curves_2d_input_no_subject(shap_3d):
    values, _, times, features = shap_3d
    flat = values[0]
    disp = ts.plot.shap.curves(flat, times, features, top_k=2)
    assert len(disp.lines) == 2


def test_curves_3d_requires_subject(shap_3d):
    values, _, times, features = shap_3d
    with pytest.raises(ValueError, match="subject"):
        ts.plot.shap.curves(values, times, features)


def test_curves_2d_rejects_subject(shap_3d):
    values, _, times, features = shap_3d
    with pytest.raises(ValueError, match="2D"):
        ts.plot.shap.curves(values[0], times, features, subject=0)


def test_curves_reference_at_zero(shap_3d):
    values, _, times, features = shap_3d
    disp = ts.plot.shap.curves(values, times, features, subject=0)
    assert disp.reference.get_ydata()[0] == pytest.approx(0.0)


def test_curves_top_k_picks_largest_max_abs(shap_3d):
    values, _, times, features = shap_3d
    # B and C have biggest peaks in our fixture
    disp = ts.plot.shap.curves(values, times, features, subject=0, top_k=2)
    assert set(disp.lines.keys()) == {"B", "C"}


def test_curves_show_others_false_yields_no_faded(shap_3d):
    values, _, times, features = shap_3d
    disp = ts.plot.shap.curves(
        values, times, features, subject=0, top_k=2, show_others=False
    )
    assert disp.faded_lines == []


def test_curves_shape_mismatch_raises(shap_3d):
    values, _, times, _ = shap_3d
    with pytest.raises(ValueError, match="features length"):
        ts.plot.shap.curves(values, times, ["A", "B"], subject=0)


def test_curves_times_mismatch_raises(shap_3d):
    values, _, _, features = shap_3d
    with pytest.raises(ValueError, match="time dimension"):
        ts.plot.shap.curves(values, np.array([1.0, 2.0]), features, subject=0)


def test_decomposition_returns_display(shap_3d):
    values, baseline, times, features = shap_3d
    disp = ts.plot.shap.local_decomposition(
        values,
        baseline,
        times,
        features,
        subject=0,
        top_k=3,
    )
    assert disp.fig is not None
    assert disp.ax is not None
    assert disp.ax_curve is not None
    assert disp.baseline_line is not None
    assert disp.prediction_line is not None
    assert len(disp.bands) > 0
    assert disp.delta_line is not None


def test_decomposition_predicted_equals_baseline_plus_sum(shap_3d):
    values, baseline, times, features = shap_3d
    disp = ts.plot.shap.local_decomposition(
        values,
        baseline,
        times,
        features,
        subject=0,
        top_k=5,
    )
    pred_y = disp.prediction_line.get_ydata()
    expected = baseline + values[0].sum(axis=0)
    np.testing.assert_allclose(pred_y, expected)


def test_decomposition_baseline_line_matches_input(shap_3d):
    values, baseline, times, features = shap_3d
    disp = ts.plot.shap.local_decomposition(
        values,
        baseline,
        times,
        features,
        subject=0,
    )
    np.testing.assert_allclose(disp.baseline_line.get_ydata(), baseline)


def test_decomposition_delta_line_matches_sum(shap_3d):
    values, baseline, times, features = shap_3d
    disp = ts.plot.shap.local_decomposition(
        values,
        baseline,
        times,
        features,
        subject=0,
    )
    np.testing.assert_allclose(
        disp.delta_line.get_ydata(),
        values[0].sum(axis=0),
    )


def test_decomposition_show_total_false_suppresses_line(shap_3d):
    values, baseline, times, features = shap_3d
    disp = ts.plot.shap.local_decomposition(
        values,
        baseline,
        times,
        features,
        subject=0,
        show_total=False,
    )
    assert disp.delta_line is None


def test_decomposition_top_k_limits_named_bands(shap_3d):
    values, baseline, times, features = shap_3d
    disp = ts.plot.shap.local_decomposition(
        values,
        baseline,
        times,
        features,
        subject=0,
        top_k=2,
        show_other=True,
    )
    # Top 2 are kept by name; rest collapse into "Other"
    named = {k for k in disp.bands.keys() if k != "Other"}
    assert len(named) <= 2
    assert "Other" in disp.bands


def test_decomposition_show_other_false_no_other_band(shap_3d):
    values, baseline, times, features = shap_3d
    disp = ts.plot.shap.local_decomposition(
        values,
        baseline,
        times,
        features,
        subject=0,
        top_k=2,
        show_other=False,
    )
    assert "Other" not in disp.bands


def test_decomposition_shape_mismatch_raises(shap_3d):
    values, baseline, times, features = shap_3d
    with pytest.raises(ValueError, match="must agree"):
        ts.plot.shap.local_decomposition(
            values,
            baseline[:5],
            times,
            features,
            subject=0,
        )


def test_decomposition_features_length_mismatch_raises(shap_3d):
    values, baseline, times, _ = shap_3d
    with pytest.raises(ValueError, match="features length"):
        ts.plot.shap.local_decomposition(
            values,
            baseline,
            times,
            ["A", "B"],
            subject=0,
        )


def test_heatmap_returns_display(shap_3d):
    values, _, times, features = shap_3d
    disp = ts.plot.shap.feature_time_heatmap(values, times, features)
    assert disp.fig is not None
    assert disp.ax is not None
    assert disp.mesh is not None
    assert disp.colorbar is not None
    assert disp.feature_order == sorted(
        features,
        key=lambda f: -np.sum(np.abs(values.mean(axis=0)[features.index(f)])),
    )


def test_heatmap_abs_mean_matches_aggregation(shap_3d):
    values, _, times, features = shap_3d
    disp = ts.plot.shap.feature_time_heatmap(
        values,
        times,
        features,
        aggregate="abs_mean",
        sort_by="name",
    )
    expected = np.mean(np.abs(values), axis=0)
    expected_sorted = expected[np.argsort(features)]
    np.testing.assert_allclose(disp.mesh.get_array(), expected_sorted)


def test_heatmap_signed_mean_matches_aggregation(shap_3d):
    values, _, times, features = shap_3d
    disp = ts.plot.shap.feature_time_heatmap(
        values,
        times,
        features,
        aggregate="signed_mean",
        sort_by="name",
    )
    expected = np.mean(values, axis=0)
    expected_sorted = expected[np.argsort(features)]
    np.testing.assert_allclose(disp.mesh.get_array(), expected_sorted)


def test_heatmap_top_k_limits_rows(shap_3d):
    values, _, times, features = shap_3d
    disp = ts.plot.shap.feature_time_heatmap(
        values,
        times,
        features,
        top_k=2,
    )
    assert len(disp.feature_order) == 2


def test_heatmap_sort_by_total_orders_by_importance(shap_3d):
    values, _, times, features = shap_3d
    disp = ts.plot.shap.feature_time_heatmap(
        values,
        times,
        features,
        sort_by="total",
    )
    # B and C have biggest absolute integrals; first two rows are them
    assert disp.feature_order[0] in {"B", "C"}
    assert disp.feature_order[1] in {"B", "C"}


def test_heatmap_sort_by_name(shap_3d):
    values, _, times, features = shap_3d
    disp = ts.plot.shap.feature_time_heatmap(
        values,
        times,
        features,
        sort_by="name",
    )
    assert disp.feature_order == sorted(features)


def test_heatmap_signed_default_cmap_is_diverging(shap_3d):
    values, _, times, features = shap_3d
    disp = ts.plot.shap.feature_time_heatmap(
        values,
        times,
        features,
        aggregate="signed_mean",
    )
    assert disp.mesh.cmap.name == "RdBu_r"


def test_heatmap_abs_default_cmap_is_sequential(shap_3d):
    values, _, times, features = shap_3d
    disp = ts.plot.shap.feature_time_heatmap(
        values,
        times,
        features,
        aggregate="abs_mean",
    )
    assert disp.mesh.cmap.name == "viridis"


def test_heatmap_colorbar_suppressible(shap_3d):
    values, _, times, features = shap_3d
    disp = ts.plot.shap.feature_time_heatmap(
        values,
        times,
        features,
        colorbar=False,
    )
    assert disp.colorbar is None


def test_heatmap_2d_input(shap_3d):
    values, _, times, features = shap_3d
    pre_aggregated = np.mean(np.abs(values), axis=0)
    disp = ts.plot.shap.feature_time_heatmap(
        pre_aggregated,
        times,
        features,
        sort_by="name",
    )
    np.testing.assert_allclose(
        disp.mesh.get_array(),
        pre_aggregated[np.argsort(features)],
    )


def test_heatmap_unknown_aggregate_raises(shap_3d):
    values, _, times, features = shap_3d
    with pytest.raises(ValueError, match="unknown aggregate"):
        ts.plot.shap.feature_time_heatmap(
            values,
            times,
            features,
            aggregate="bogus",
        )


def test_heatmap_unknown_sort_by_raises(shap_3d):
    values, _, times, features = shap_3d
    with pytest.raises(ValueError, match="unknown sort_by"):
        ts.plot.shap.feature_time_heatmap(
            values,
            times,
            features,
            sort_by="bogus",
        )


def test_heatmap_top_k_zero_raises(shap_3d):
    values, _, times, features = shap_3d
    with pytest.raises(ValueError, match="positive"):
        ts.plot.shap.feature_time_heatmap(
            values,
            times,
            features,
            top_k=0,
        )
