"""Catalogue overview for the datasets page.

One dot per open study (variants left out, so each cohort appears once):
subjects against event rate, coloured by the number of competing causes.
The first build downloads every open dataset; later builds read the cache.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

import tausurv as ts

LABELLED = (
    "aml",
    "flchain",
    "gbsg",
    "metabric",
    "nafld",
    "nwtco",
    "pbc",
    "prostate",
    "rotterdam",
    "support",
)
CAUSE_LABELS = {1: "single risk", 2: "2 causes", 3: "3 causes"}


def make(out_path: Path) -> None:
    ts.plot.set_style("publication")
    names = ts.datasets.list_datasets(base_only=True, access="open")
    bunches = [ts.datasets.load_dataset(name) for name in names]
    n_subjects = np.array([len(b.event_time) for b in bunches])
    event_rate = np.array([np.mean(b.event_indicator > 0) for b in bunches])
    n_causes = np.array([b.n_causes for b in bunches])

    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    for k, (causes, label) in enumerate(CAUSE_LABELS.items()):
        mask = n_causes == causes
        ax.scatter(
            n_subjects[mask],
            event_rate[mask],
            s=22,
            color=ts.plot.colors.OKABE_ITO[k],
            label=label,
            zorder=3,
        )
    for name, x, y in zip(names, n_subjects, event_rate, strict=True):
        if name in LABELLED:
            ax.annotate(
                name,
                (x, y),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=9,
            )

    ax.set_xscale("log")
    ax.set_ylim(0, 1.03)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlabel("subjects")
    ax.set_ylabel("event rate")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out = (
        Path(__file__).resolve().parent.parent
        / "public"
        / "figures"
        / "dataset_catalogue.svg"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    make(out)
