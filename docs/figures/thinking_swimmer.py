"""Six patients followed for up to five years, each ending in a different fate.

Used in the "Thinking in Survival" concept page to introduce the (Y, delta)
data layout, the at-risk set, and the visual taxonomy of censoring.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from _style import ACCENT, INK, MUTED, setup_style


def make(out_path: Path) -> None:
    setup_style()

    subjects = [
        ("A", 1.0, "event"),
        ("B", 1.8, "lost"),
        ("C", 3.0, "event"),
        ("D", 4.0, "withdrew"),
        ("E", 4.5, "event"),
        ("F", 5.0, "admin"),
    ]
    n = len(subjects)
    end_of_study = 5.0

    fig, ax = plt.subplots(figsize=(7.2, 3.2))

    ax.axvline(
        end_of_study,
        color=MUTED,
        linestyle=(0, (4, 3)),
        linewidth=1.0,
        alpha=0.7,
        zorder=1,
    )
    ax.text(
        end_of_study + 0.08,
        (n + 1) / 2,
        "end of study",
        color=MUTED,
        rotation=90,
        ha="center",
        va="center",
        fontsize=10,
    )

    for idx, (_label, y_i, kind) in enumerate(subjects):
        row = idx + 1
        ax.plot([0, y_i], [row, row], color=MUTED, linewidth=1.5, zorder=2)
        if kind == "event":
            ax.plot(
                y_i,
                row,
                "o",
                markerfacecolor=ACCENT,
                markeredgecolor=ACCENT,
                markersize=9,
                zorder=3,
            )
            ax.text(
                y_i + 0.18,
                row,
                "event",
                color=INK,
                fontweight="bold",
                va="center",
                fontsize=11,
            )
        elif kind == "admin":
            ax.plot(
                y_i,
                row,
                "|",
                color=MUTED,
                markersize=12,
                markeredgewidth=2.0,
                zorder=3,
            )
        else:
            ax.plot(
                y_i,
                row,
                ">",
                markerfacecolor="white",
                markeredgecolor=MUTED,
                markersize=8,
                markeredgewidth=1.5,
                zorder=3,
            )
            ax.text(
                y_i + 0.18,
                row,
                kind,
                color=MUTED,
                va="center",
                fontsize=11,
            )

    ax.set_yticks(range(1, n + 1))
    ax.set_yticklabels([s[0] for s in subjects])
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xlim(-0.1, 5.15)
    ax.set_ylim(0.4, n + 0.6)
    ax.invert_yaxis()
    ax.set_xlabel("time in years")
    ax.tick_params(axis="y", length=0)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    out = (
        Path(__file__).resolve().parent.parent
        / "public"
        / "figures"
        / "thinking_swimmer.svg"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    make(out)
