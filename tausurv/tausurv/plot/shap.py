r"""SHAP visualization plots for time-varying survival explanations.

Three plots, each answering a question general-purpose SHAP libraries do not:

- :func:`curves` -- per-subject SHAP-over-time, one line per feature.
  Answers *when* each feature matters and whether contributions flip sign.
- :func:`local_decomposition` -- per-subject two-panel figure: predicted
  :math:`\hat S(t \mid x)` overlaid on a baseline curve, with a stacked
  decomposition below showing the feature-by-feature contributions that sum
  to :math:`\hat S(t \mid x) - S_0(t)`. The single plot that explains
  *why* one patient's trajectory looks the way it does.
- :func:`feature_time_heatmap` -- cohort view, features-by-times heatmap
  of aggregated SHAP. Answers *at what horizon* each feature peaks in
  importance.

The functions accept precomputed SHAP arrays -- computing SHAP itself is
not in scope here. Array conventions:

- 3-D ``(n_subjects, n_features, n_times)``: full cohort with time-varying
  attributions.
- 2-D ``(n_features, n_times)``: a single subject (or pre-aggregated cohort).

Reference: Krzyzinski et al. (2023). *SurvSHAP(t): Time-dependent
explanations of machine learning survival models.* Knowledge-Based Systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np

from tausurv.plot._primitives import reference_line

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection, QuadMesh
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from numpy.typing import ArrayLike


Aggregate = Literal["abs_mean", "signed_mean"]
SortBy = Literal["total", "peak_time", "name"]


@dataclass
class ShapCurvesDisplay:
    """Result of :func:`curves`.

    Attributes
    ----------
    fig : Figure
    ax : Axes
    lines : dict[str, Line2D]
        Top-K feature curves, keyed by feature name.
    faded_lines : list[Line2D]
        The faded background curves for features outside the top-K.
    reference : Line2D or None
        The :math:`y = 0` reference line.
    """

    fig: "Figure"
    ax: "Axes"
    lines: dict[str, "Line2D"] = field(default_factory=dict)
    faded_lines: list["Line2D"] = field(default_factory=list)
    reference: "Line2D | None" = None


@dataclass
class ShapDecompositionDisplay:
    """Result of :func:`local_decomposition`.

    Attributes
    ----------
    fig : Figure
    ax_curve : Axes
        Top panel: predicted vs baseline survival.
    ax : Axes
        Bottom panel: stacked feature decomposition.
    baseline_line : Line2D
    prediction_line : Line2D
    bands : dict[str, list[PolyCollection]]
        Stacked-area artists keyed by feature name. A feature with both
        positive and negative contributions across time appears twice in
        its list (one band above zero, one below).
    delta_line : Line2D or None
        The :math:`\\hat S(t \\mid x) - S_0(t)` reference line drawn on the
        bottom panel, when ``show_total=True``.
    """

    fig: "Figure"
    ax_curve: "Axes"
    ax: "Axes"
    baseline_line: "Line2D | None" = None
    prediction_line: "Line2D | None" = None
    bands: dict[str, list["PolyCollection"]] = field(default_factory=dict)
    delta_line: "Line2D | None" = None


@dataclass
class ShapHeatmapDisplay:
    """Result of :func:`feature_time_heatmap`.

    Attributes
    ----------
    fig : Figure
    ax : Axes
    mesh : QuadMesh
    colorbar : Colorbar or None
    feature_order : list[str]
        The rendered row ordering, top to bottom.
    """

    fig: "Figure"
    ax: "Axes"
    mesh: "QuadMesh | None" = None
    colorbar: "Colorbar | None" = None
    feature_order: list[str] = field(default_factory=list)


def curves(
    values: "ArrayLike",
    times: "ArrayLike",
    features: "Sequence[str]",
    *,
    subject: int | None = None,
    top_k: int = 8,
    ax: "Axes | None" = None,
    direct_labels: bool = True,
    legend: bool = False,
    show_others: bool = True,
    xlabel: str = "Time",
    ylabel: str = "SHAP",
    title: str | None = None,
) -> ShapCurvesDisplay:
    r"""SHAP-over-time curves for a single subject.

    Plots :math:`\mathrm{SHAP}_j(t)` as a function of time, one line per
    feature. The top-K most important features are drawn in the palette
    foreground; the rest are drawn as faded gray lines for context (toggle
    with ``show_others``).

    Parameters
    ----------
    values : 2D or 3D array
        ``(n_features, n_times)`` for a single subject, or
        ``(n_subjects, n_features, n_times)`` with ``subject=`` index.
    times : (n_times,) array
    features : sequence of str
    subject : int, optional
        Required when ``values`` is 3D.
    top_k : int, default 8
        Number of foreground features. Selection by max :math:`|\mathrm{SHAP}|`
        across time.
    ax : Axes, optional
    direct_labels : bool, default True
        Label foreground curves at their right endpoint instead of a legend.
    legend : bool, default False
        Show a legend instead. Ignored when ``direct_labels=True``.
    show_others : bool, default True
        Draw faded background curves for the remaining features.
    xlabel, ylabel, title : str

    Returns
    -------
    ShapCurvesDisplay
    """
    import matplotlib.pyplot as plt

    arr = _select_2d(values, subject, where="values")
    t = _check_1d(times, "times")
    if arr.shape[1] != t.size:
        raise ValueError(
            f"values' time dimension ({arr.shape[1]}) must match times "
            f"length ({t.size})"
        )
    if len(features) != arr.shape[0]:
        raise ValueError(
            f"features length {len(features)} must match values' feature "
            f"dimension {arr.shape[0]}"
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be positive; got {top_k}")

    importance = np.max(np.abs(arr), axis=1)
    order = np.argsort(-importance)
    top = order[: min(top_k, len(features))]
    others = order[top_k:]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    disp = ShapCurvesDisplay(fig=fig, ax=ax)
    disp.reference = reference_line(ax, 0.0)

    if show_others:
        for idx in others:
            (line,) = ax.plot(
                t, arr[idx],
                color="#cccccc", linewidth=0.6, alpha=0.55, zorder=1,
            )
            disp.faded_lines.append(line)

    for idx in top:
        name = features[idx]
        kw: dict = {"linewidth": 1.8, "zorder": 2}
        if not direct_labels and name:
            kw["label"] = name
        (line,) = ax.plot(t, arr[idx], **kw)
        disp.lines[name] = line

    if direct_labels:
        _direct_label_at_end(ax, t[-1], disp.lines, arr, top, features)
    elif legend and len(disp.lines) >= 2:
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    return disp


def local_decomposition(
    values: "ArrayLike",
    baseline_survival: "ArrayLike",
    times: "ArrayLike",
    features: "Sequence[str]",
    *,
    subject: int | None = None,
    top_k: int = 6,
    fig: "Figure | None" = None,
    figsize: "tuple[float, float] | None" = None,
    diverging_cmap: str = "RdBu",
    show_total: bool = True,
    show_other: bool = True,
    legend: bool = True,
    xlabel: str = "Time",
    title: str | None = None,
) -> ShapDecompositionDisplay:
    r"""Per-subject survival decomposition: prediction = baseline + sum of SHAPs.

    Two-panel figure with a shared time axis:

    - **Top:** baseline :math:`S_0(t)` (gray dashed) and the predicted
      :math:`\hat S(t \mid x_i) = S_0(t) + \sum_j \mathrm{SHAP}_j(t)` (solid).
      The visible vertical gap *is* the total SHAP effect on survival.
    - **Bottom:** stacked feature contributions. Positive contributions
      stack upward from zero, negative contributions stack downward.
      Colour for each feature comes from a diverging palette ordered by the
      feature's mean signed contribution -- "this feature mostly increases
      survival" reads as blue, "mostly decreases" reads as red. Remaining
      features outside ``top_k`` collapse into an ``"Other"`` band.

    Parameters
    ----------
    values : 2D or 3D array
        Contributions to :math:`\hat S(t \mid x)` (not log-risk). Shape
        ``(n_features, n_times)`` or ``(n_subjects, n_features, n_times)``.
        :math:`\sum_j \mathrm{SHAP}_j(t)` should equal
        :math:`\hat S(t \mid x) - S_0(t)` at every ``t``.
    baseline_survival : (n_times,) array
    times : (n_times,) array
    features : sequence of str
    subject : int, optional
    top_k : int, default 6
    fig : Figure, optional
        Existing figure to draw into. When ``None`` a fresh figure is made.
    figsize : (float, float), optional
        Used only when ``fig`` is created here.
    diverging_cmap : str, default "RdBu_r"
        Drives the per-feature colour assignment.
    show_total : bool, default True
        Overlay the :math:`\Delta \hat S(t)` line on the bottom panel.
    show_other : bool, default True
        Aggregate features outside ``top_k`` into a single ``"Other"`` band
        so the visible stack always sums to :math:`\Delta \hat S(t)`.
    legend : bool, default True
        Legend on the bottom panel.
    xlabel, title : str

    Returns
    -------
    ShapDecompositionDisplay
    """
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    arr = _select_2d(values, subject, where="values")
    t = _check_1d(times, "times")
    baseline = _check_1d(baseline_survival, "baseline_survival")
    if arr.shape[1] != t.size or baseline.size != t.size:
        raise ValueError(
            f"values' time dimension, baseline length, and times length "
            f"must agree; got {arr.shape[1]}, {baseline.size}, {t.size}"
        )
    if len(features) != arr.shape[0]:
        raise ValueError(
            f"features length {len(features)} must match values' feature "
            f"dimension {arr.shape[0]}"
        )

    if fig is None:
        fig = plt.figure(figsize=figsize or (7.0, 5.0))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[3.0, 4.5], hspace=0.08,
        left=0.10, right=0.80 if legend else 0.96, top=0.92, bottom=0.10,
    )
    ax_curve = fig.add_subplot(gs[0])
    ax_decomp = fig.add_subplot(gs[1], sharex=ax_curve)

    delta = arr.sum(axis=0)
    predicted = baseline + delta

    (baseline_line,) = ax_curve.plot(
        t, baseline, color="#888888", linestyle="--", linewidth=0.8,
        label="Baseline",
    )
    (prediction_line,) = ax_curve.plot(
        t, predicted, color="#0072B2", linewidth=1.8,
        label="Predicted",
    )
    ax_curve.set_ylabel(r"$\hat S(t)$")
    ax_curve.set_ylim(0.0, 1.02)
    ax_curve.tick_params(axis="x", labelbottom=False)
    ax_curve.legend(loc="lower left", frameon=False)
    if title is not None:
        ax_curve.set_title(title)

    importance = np.max(np.abs(arr), axis=1)
    order_all = np.argsort(-importance)
    top = order_all[: min(top_k, len(features))]
    other = order_all[top_k:]

    cmap = colormaps[diverging_cmap]
    mean_signed = arr[top].mean(axis=1)
    max_abs = float(np.max(np.abs(mean_signed))) if mean_signed.size else 1.0
    if max_abs == 0.0:
        max_abs = 1.0
    feature_colors = {
        features[idx]: cmap(0.5 + 0.5 * (mean_signed[k] / max_abs))
        for k, idx in enumerate(top)
    }

    pos_stack = np.zeros_like(t)
    neg_stack = np.zeros_like(t)
    disp = ShapDecompositionDisplay(
        fig=fig, ax_curve=ax_curve, ax=ax_decomp,
        baseline_line=baseline_line, prediction_line=prediction_line,
    )

    for idx in top:
        name = features[idx]
        cj = arr[idx]
        col = feature_colors[name]
        bands = disp.bands.setdefault(name, [])

        pos_j = np.where(cj > 0, cj, 0.0)
        if pos_j.any():
            poly = ax_decomp.fill_between(
                t, pos_stack, pos_stack + pos_j,
                color=col, linewidth=0, label=name,
            )
            bands.append(poly)
        pos_stack = pos_stack + pos_j

        neg_j = np.where(cj < 0, cj, 0.0)
        if neg_j.any():
            label = None if name in {b.get_label() for b in bands} else name
            poly = ax_decomp.fill_between(
                t, neg_stack, neg_stack + neg_j,
                color=col, linewidth=0,
                **({"label": label} if label and not pos_j.any() else {}),
            )
            bands.append(poly)
        neg_stack = neg_stack + neg_j

    if show_other and other.size:
        other_total = arr[other].sum(axis=0)
        pos_other = np.where(other_total > 0, other_total, 0.0)
        neg_other = np.where(other_total < 0, other_total, 0.0)
        col = "#cccccc"
        bands = disp.bands.setdefault("Other", [])
        if pos_other.any():
            poly = ax_decomp.fill_between(
                t, pos_stack, pos_stack + pos_other,
                color=col, linewidth=0, label=f"Other ({other.size})",
            )
            bands.append(poly)
            pos_stack = pos_stack + pos_other
        if neg_other.any():
            poly = ax_decomp.fill_between(
                t, neg_stack, neg_stack + neg_other,
                color=col, linewidth=0,
                **(
                    {"label": f"Other ({other.size})"}
                    if not pos_other.any() else {}
                ),
            )
            bands.append(poly)
            neg_stack = neg_stack + neg_other

    ax_decomp.axhline(0.0, color="#444444", linewidth=0.6, zorder=2)
    if show_total:
        (disp.delta_line,) = ax_decomp.plot(
            t, delta, color="#111111", linewidth=1.6, zorder=4,
            label=r"$\Delta \hat S(t)$",
        )

    ax_decomp.set_xlabel(xlabel)
    ax_decomp.set_ylabel(r"$\Delta \hat S(t)$ contribution")
    if legend:
        ax_decomp.legend(
            loc="upper left", bbox_to_anchor=(1.01, 1.0),
            fontsize=8, frameon=False, borderaxespad=0.0,
        )

    return disp


def feature_time_heatmap(
    values: "ArrayLike",
    times: "ArrayLike",
    features: "Sequence[str]",
    *,
    aggregate: Aggregate = "abs_mean",
    sort_by: SortBy = "total",
    top_k: int | None = None,
    ax: "Axes | None" = None,
    colorbar: bool = True,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    xlabel: str = "Time",
    ylabel: str = "Feature",
    title: str | None = None,
) -> ShapHeatmapDisplay:
    r"""Cohort-level features-by-times SHAP heatmap.

    Aggregates a 3-D SHAP array over subjects to produce one value per
    ``(feature, time)`` cell. Rows are sorted by global importance so the
    top of the heatmap is "the features that matter most across the time
    horizon studied".

    Parameters
    ----------
    values : 2D or 3D array
        ``(n_subjects, n_features, n_times)``; or 2D ``(n_features, n_times)``
        if already aggregated.
    times : (n_times,) array
    features : sequence of str
    aggregate : {"abs_mean", "signed_mean"}, default "abs_mean"
        How to collapse the subject axis. ``"abs_mean"`` is the
        magnitude / importance view (sequential cmap by default);
        ``"signed_mean"`` is the direction view (diverging cmap by default).
    sort_by : {"total", "peak_time", "name"}, default "total"
        Row ordering. ``"total"``: sum of |SHAP| across times.
        ``"peak_time"``: time of maximum |SHAP|. ``"name"``: alphabetical.
    top_k : int, optional
        Show only this many top rows.
    ax : Axes, optional
    colorbar : bool, default True
    cmap : str, optional
        Defaults to ``"viridis"`` for ``abs_mean`` and ``"RdBu_r"`` for
        ``signed_mean``.
    vmin, vmax : float, optional
        Manual colour-scale bounds.
    xlabel, ylabel, title : str

    Returns
    -------
    ShapHeatmapDisplay
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 3:
        if aggregate == "abs_mean":
            agg = np.mean(np.abs(arr), axis=0)
        elif aggregate == "signed_mean":
            agg = np.mean(arr, axis=0)
        else:
            raise ValueError(
                f"unknown aggregate {aggregate!r}; expected 'abs_mean' or "
                "'signed_mean'"
            )
    elif arr.ndim == 2:
        agg = arr
    else:
        raise ValueError(
            f"values must be 2D or 3D; got shape {arr.shape}"
        )

    t = _check_1d(times, "times")
    if agg.shape[1] != t.size:
        raise ValueError(
            f"values' time dimension ({agg.shape[1]}) must match times "
            f"length ({t.size})"
        )
    if len(features) != agg.shape[0]:
        raise ValueError(
            f"features length {len(features)} must match values' feature "
            f"dimension {agg.shape[0]}"
        )

    if sort_by == "total":
        order = np.argsort(-np.sum(np.abs(agg), axis=1))
    elif sort_by == "peak_time":
        order = np.argsort(np.argmax(np.abs(agg), axis=1))
    elif sort_by == "name":
        order = np.argsort(features)
    else:
        raise ValueError(
            f"unknown sort_by {sort_by!r}; expected 'total', 'peak_time', "
            "or 'name'"
        )

    if top_k is not None:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive; got {top_k}")
        order = order[:top_k]

    sorted_agg = agg[order]
    sorted_names = [features[i] for i in order]

    if cmap is None:
        cmap = "viridis" if aggregate == "abs_mean" else "RdBu_r"

    if aggregate == "signed_mean" and vmin is None and vmax is None:
        extent = float(np.max(np.abs(sorted_agg)))
        norm = Normalize(vmin=-extent, vmax=extent)
    else:
        norm = Normalize(
            vmin=vmin if vmin is not None else float(np.min(sorted_agg)),
            vmax=vmax if vmax is not None else float(np.max(sorted_agg)),
        )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    disp = ShapHeatmapDisplay(fig=fig, ax=ax, feature_order=sorted_names)
    y = np.arange(len(sorted_names))
    disp.mesh = ax.pcolormesh(
        t, y, sorted_agg, cmap=cmap, norm=norm, shading="nearest",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    if title is not None:
        ax.set_title(title)

    if colorbar:
        disp.colorbar = fig.colorbar(disp.mesh, ax=ax, pad=0.02, fraction=0.046)
        disp.colorbar.set_label(
            "Mean |SHAP|" if aggregate == "abs_mean" else "Mean SHAP",
        )

    return disp


def _select_2d(
    values: "ArrayLike", subject: int | None, where: str,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 3:
        if subject is None:
            raise ValueError(
                f"{where} is 3D (n_subjects, n_features, n_times); subject= "
                "is required to select one"
            )
        if not -arr.shape[0] <= subject < arr.shape[0]:
            raise ValueError(
                f"subject index {subject} out of range for "
                f"{arr.shape[0]} subjects"
            )
        return arr[subject]
    if arr.ndim == 2:
        if subject is not None:
            raise ValueError(
                f"{where} is 2D; subject= is not valid (already a single "
                "subject)"
            )
        return arr
    raise ValueError(
        f"{where} must be 2D or 3D; got shape {arr.shape}"
    )


def _check_1d(x: "ArrayLike", where: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{where} must be 1D; got shape {a.shape}")
    return a


def _direct_label_at_end(
    ax: "Axes",
    x_end: float,
    lines: dict[str, "Line2D"],
    arr: np.ndarray,
    indices: np.ndarray,
    features: "Sequence[str]",
) -> None:
    for idx in indices:
        name = features[idx]
        line = lines[name]
        ax.annotate(
            name,
            xy=(x_end, arr[idx, -1]),
            xytext=(4, 0), textcoords="offset points",
            fontsize=8, color=line.get_color(),
            va="center", ha="left",
        )
