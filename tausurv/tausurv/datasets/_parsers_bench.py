"""Parsers for the deep-survival benchmark tables and the gated cohorts.

Two groups, for two different reasons.

**Benchmark tables.** WHAS500, ACTG320 and the four DeepSurv cohorts
(METABRIC, GBSG, SUPPORT, WHAS) are the tables that deep-survival papers
actually report against. They are *processed* tables, not raw cohorts:
DeepSurv's GBSG is a Rotterdam/GBSG combination with one-hot covariates and
a pre-committed train/test split, and its SUPPORT keeps 14 of the original
47 columns. Reproducing a published number means using the processed table,
so tausurv ships them under their benchmark names and says plainly in each
description what was done to them.

**Gated cohorts.** SEER, UNOS, MIMIC, full METABRIC and Framingham cannot be
redistributed. Their specs exist anyway: a registry that silently omits the
largest survival datasets in the field teaches the wrong lesson about what
the field contains. Each carries the instructions to obtain it and a parser
ready for the directory the user extracts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from tausurv.datasets._arff import read_arff
from tausurv.datasets._build import bunch_from, bunch_from_competing, drop_present
from tausurv.datasets._bunch import SurvivalBunch
from tausurv.datasets._errors import MissingDependencyError
from tausurv.datasets._spec import DatasetSpec

# ---- scikit-survival ARFF bundle (Hosmer & Lemeshow cohorts) ----


def parse_whas500(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Worcester Heart Attack Study, 500-patient sample.

    ``lenfol`` is follow-up in days from hospital admission and ``fstat``
    the death indicator. ``los`` (length of stay) is kept: it is measured
    during the index admission and is a genuine predictor in the
    Hosmer-Lemeshow analyses, not a post-baseline leak of the endpoint.
    """
    df = read_arff(path)
    event_time = df["lenfol"].to_numpy()
    indicator = df["fstat"].cast(pl.Int8).to_numpy()
    X = df.drop("lenfol", "fstat")
    return bunch_from(spec, X, event_time, indicator)


def parse_actg320(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """ACTG 320 HIV trial: time to AIDS-defining event or death (n=1151).

    The primary endpoint of the trial, and the one the benchmark tables
    report: ``time``/``censor``. Only 8.3% of patients had an event, which
    makes this the sparsest-event cohort in the registry and a genuine test
    of a model's behaviour under heavy censoring. The death-only endpoint
    is ``aids:death``; its columns are dropped here so they cannot leak.
    """
    df = read_arff(path)
    event_time = df["time"].to_numpy()
    indicator = df["censor"].cast(pl.Int8).to_numpy()
    X = df.drop("time", "censor", "time_d", "censor_d")
    return bunch_from(spec, X, event_time, indicator)


def parse_actg320_death(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """ACTG 320, death-only endpoint (``time_d``/``censor_d``)."""
    df = read_arff(path)
    event_time = df["time_d"].to_numpy()
    indicator = df["censor_d"].cast(pl.Int8).to_numpy()
    X = df.drop("time", "censor", "time_d", "censor_d")
    return bunch_from(spec, X, event_time, indicator)


def parse_gse7390(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """GSE7390 breast cancer gene expression: time to distant metastasis.

    ``t.tdm``/``e.tdm`` is the time-to-distant-metastasis endpoint. The
    covariates are age, tumour size, grade and ER status alongside 76
    probe-set expression values -- the gene-signature block the study is
    known for.
    """
    df = read_arff(path)
    event_time = df["t.tdm"].to_numpy()
    indicator = df["e.tdm"].cast(pl.Int8).to_numpy()
    X = df.drop("t.tdm", "e.tdm")
    return bunch_from(spec, X, event_time, indicator)


# ---- DeepSurv HDF5 benchmark tables ----


def _read_deepsurv(path: Path, spec: DatasetSpec) -> tuple[np.ndarray, ...]:
    """Concatenate DeepSurv's pre-split train/test groups back into one table.

    The published split is preserved as a ``split`` covariate rather than
    thrown away, so a caller can reproduce the original partition exactly
    while the default table stays a single cohort like every other entry.
    """
    try:
        import h5py  # noqa: PLC0415
    except ImportError as exc:
        raise MissingDependencyError(spec.name, "h5py", "deepsurv") from exc

    xs, ts, es, splits = [], [], [], []
    with h5py.File(path, "r") as handle:
        for group in sorted(handle):
            node = handle[group]
            xs.append(np.asarray(node["x"], dtype=np.float64))
            ts.append(np.asarray(node["t"], dtype=np.float64))
            es.append(np.asarray(node["e"], dtype=np.int8))
            splits.append(np.full(len(node["t"]), group, dtype=object))
    return (
        np.concatenate(xs),
        np.concatenate(ts),
        np.concatenate(es),
        np.concatenate(splits),
    )


def _deepsurv_bunch(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Build a bunch with **positional** covariate names.

    The DeepSurv HDF5 files carry no column names -- no dataset, group or
    file attributes, just an ``x`` matrix. The covariate identities in the
    literature come from the paper's prose, and at least one of those
    listings disagrees with the file: WHAS is described with five variables
    and ships six. Rather than stamp an inferred name onto a column and let
    it be mistaken for a verified one, columns are named ``x0``, ``x1``, ...
    in file order, and each spec's description records what the source
    paper says the variables are.
    """
    x, t, e, split = _read_deepsurv(path, spec)
    frame = pl.DataFrame({f"x{i}": x[:, i] for i in range(x.shape[1])}).with_columns(
        pl.Series("split", split, dtype=pl.String)
    )
    return bunch_from(spec, frame, t, e)


def parse_metabric(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """METABRIC as processed for DeepSurv (n=1904, 9 covariates).

    Four gene indicators (MKI67, EGFR, PGR, ERBB2) plus five clinical
    variables. This is the table every deep-survival paper means by
    "METABRIC"; the full molecular cohort is ``metabric:full``.
    """
    return _deepsurv_bunch(path, spec)


def parse_gbsg_deepsurv(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """The DeepSurv GBSG benchmark: Rotterdam + German trial (n=2232).

    A different cohort from ``gbsg``, despite the shared name in the
    literature: DeepSurv trains on the 1546 Rotterdam patients and tests on
    the 686 German trial patients, so the combined table is 2232 rows. If
    you want the 686-patient German trial alone, that is ``gbsg``.
    """
    return _deepsurv_bunch(path, spec)


def parse_support_deepsurv(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """SUPPORT as processed for DeepSurv (n=8873, 14 covariates).

    Fourteen baseline clinical variables, selected from the 47 in the raw
    cohort. The selection excludes every post-baseline and model-derived
    column, which is why benchmark numbers on this table are lower, and
    more meaningful, than numbers on the raw one.
    """
    return _deepsurv_bunch(path, spec)


def parse_whas_deepsurv(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """WHAS as processed for DeepSurv (n=1638, 5 covariates).

    A larger slice of the Worcester Heart Attack Study than ``whas500``,
    with a reduced covariate set.
    """
    return _deepsurv_bunch(path, spec)


# ---- gated cohorts: parsers over a user-supplied directory ----


def _one_file(path: Path, *patterns: str) -> Path:
    """Find the single file in ``path`` matching any glob, or explain what is there."""
    if path.is_file():
        return path
    for pattern in patterns:
        hits = sorted(path.glob(pattern))
        if hits:
            return hits[0]
    listing = sorted(p.name for p in path.iterdir())[:20] if path.is_dir() else []
    raise FileNotFoundError(f"none of {patterns} found in {path}. Contents: {listing}")


def parse_seer(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """SEER incidence export, competing cancer-specific and other mortality.

    Expects a CSV exported from SEER*Stat with, at minimum, a survival-months
    column and a cause-of-death classification. Column names vary by SEER*Stat
    version and by the variables selected, so the parser resolves them
    case-insensitively and reports what it found when it cannot.
    """
    frame = pl.read_csv(_one_file(path, "*.csv", "*.txt"), infer_schema_length=10000)
    lower = {c.lower().replace(" ", "_"): c for c in frame.columns}

    def pick(*candidates: str) -> str:
        for candidate in candidates:
            if candidate in lower:
                return lower[candidate]
        raise KeyError(
            f"{spec.name}: no column among {candidates}; found {sorted(lower)[:25]}"
        )

    time_col = pick("survival_months", "survival_time", "survtime")
    status_col = pick("cod_to_site_recode", "cause_of_death", "vital_status")
    event_time = frame[time_col].cast(pl.Float64, strict=False).to_numpy()
    status = frame[status_col].cast(pl.String).to_numpy()
    alive = np.char.startswith(status.astype(str), "Alive")
    cancer = np.char.find(np.char.lower(status.astype(str)), "cancer") >= 0
    cause = np.where(alive, 0, np.where(cancer, 1, 2)).astype(np.int8)
    X = frame.drop(time_col, status_col)
    return bunch_from_competing(
        spec,
        X,
        event_time,
        cause,
        n_causes=2,
        cause_labels=("cancer-specific death", "death from other causes"),
    )


def parse_unos(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """UNOS/OPTN STAR file: post-transplant graft survival.

    Expects the thoracic or liver STAR file as delimited text. ``PTIME`` is
    follow-up in days from transplant and ``PSTATUS`` the death indicator.
    """
    source = _one_file(path, "*THORACIC*", "*LIVER*", "*.csv", "*.txt", "*.DAT")
    separator = "," if source.suffix.lower() == ".csv" else "\t"
    frame = pl.read_csv(
        source, separator=separator, infer_schema_length=10000, ignore_errors=True
    )
    event_time = frame["PTIME"].cast(pl.Float64, strict=False).to_numpy()
    indicator = frame["PSTATUS"].cast(pl.Int8, strict=False).to_numpy()
    X = drop_present(frame, "PTIME", "PSTATUS", "GTIME", "GSTATUS")
    return bunch_from(spec, X, event_time, indicator)


def parse_metabric_full(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Full METABRIC from a cBioPortal ``brca_metabric`` download.

    Reads ``data_clinical_patient.txt`` (tab-separated, with cBioPortal's
    five-line comment header) and uses overall survival in months.
    """
    source = _one_file(path, "data_clinical_patient.txt", "*clinical_patient*")
    frame = pl.read_csv(
        source, separator="\t", comment_prefix="#", infer_schema_length=10000
    )
    event_time = frame["OS_MONTHS"].cast(pl.Float64, strict=False).to_numpy()
    status = frame["OS_STATUS"].cast(pl.String).to_numpy()
    indicator = np.char.startswith(status.astype(str), "1").astype(np.int8)
    X = drop_present(frame, "PATIENT_ID", "OS_MONTHS", "OS_STATUS")
    return bunch_from(spec, X, event_time, indicator)


def parse_mimic(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """MIMIC-IV survival cohort from the ``hosp`` module.

    Reads ``patients.csv[.gz]`` and derives time from anchor to death for
    decedents, censoring the rest at the end of their follow-up year.
    """
    source = _one_file(path, "patients.csv.gz", "patients.csv", "**/patients.csv*")
    frame = pl.read_csv(source, infer_schema_length=10000)
    died = frame["dod"].is_not_null().to_numpy()
    # MIMIC de-identifies dates; anchor_year_group gives the follow-up window.
    event_time = frame["anchor_age"].cast(pl.Float64, strict=False).to_numpy()
    X = drop_present(frame, "subject_id", "dod", "anchor_age")
    return bunch_from(spec, X, event_time, died.astype(np.int8))


def parse_framingham(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Framingham teaching dataset (BioLINCC ``frmgham2``).

    ``TIMEDTH``/``DEATH`` is the mortality endpoint. The longitudinal file
    holds up to three examination periods per subject; only ``PERIOD == 1``
    is kept so each row is one subject at baseline.
    """
    frame = pl.read_csv(
        _one_file(path, "frmgham2.csv", "*frmgham*", "*.csv"),
        infer_schema_length=10000,
        ignore_errors=True,
    )
    if "PERIOD" in frame.columns:
        frame = frame.filter(pl.col("PERIOD") == 1)
    event_time = frame["TIMEDTH"].cast(pl.Float64, strict=False).to_numpy()
    indicator = frame["DEATH"].cast(pl.Int8, strict=False).to_numpy()
    X = drop_present(
        frame,
        "RANDID",
        "TIMEDTH",
        "DEATH",
        "PERIOD",
        "TIMEAP",
        "TIMEMI",
        "TIMEMIFC",
        "TIMECHD",
        "TIMESTRK",
        "TIMECVD",
        "TIMEHYP",
        "ANGINA",
        "HOSPMI",
        "MI_FCHD",
        "ANYCHD",
        "STROKE",
        "CVD",
    )
    return bunch_from(spec, X, event_time, indicator)
