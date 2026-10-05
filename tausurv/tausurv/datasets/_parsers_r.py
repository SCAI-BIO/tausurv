"""Parsers for the R-ecosystem cohorts served by Rdatasets.

Three source packages, each with its own habits:

``survival``
    Therneau's package, the reference implementation of most of the field.
    Endpoints are usually ``(futime, status)`` or ``(time, status)``, but
    several cohorts carry two endpoints at once (recurrence and death), which
    is where the ``:variant`` names come from.
``KMsurv``
    The datasets from Klein & Moeschberger's textbook. Mostly small, mostly
    ``(time, delta)``, and mostly the worked example someone learned survival
    analysis from.
``asaur``
    Moore's *Applied Survival Analysis Using R*.

Every Rdatasets export carries a ``rownames`` index column, which is dropped
everywhere. Nothing else is imputed, encoded or scaled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from tausurv.datasets._build import bunch_from, bunch_from_competing, drop_present
from tausurv.datasets._bunch import SurvivalBunch
from tausurv.datasets._spec import DatasetSpec


#: Rdatasets writes R's missing value as the literal string ``NA``; without
#: this, any column carrying one is either rejected or silently read as text.
_NA = ["NA", "NaN", ""]


def _read(path: Path) -> pl.DataFrame:
    """Read an Rdatasets CSV with R's missing-value convention honoured.

    A few exports (``udca1``) write the row index into an unnamed leading
    column rather than one called ``rownames``; it is dropped here so no
    parser has to know which convention its file used.
    """
    frame = pl.read_csv(path, null_values=_NA, infer_schema_length=10000)
    return frame.drop([c for c in frame.columns if not c.strip()])


def _simple(
    path: Path,
    spec: DatasetSpec,
    *,
    time: str,
    status: str,
    event_value: object = None,
    drop: tuple[str, ...] = (),
) -> SurvivalBunch:
    """The common case: one time column, one indicator, drop the rest.

    ``event_value`` recodes a non-0/1 indicator (R often uses 1=censored,
    2=dead); leave it ``None`` when the column is already 0/1.
    """
    df = _read(path)
    event_time = df[time].cast(pl.Float64).to_numpy()
    if event_value is None:
        indicator = df[status].cast(pl.Int8).to_numpy()
    else:
        indicator = (df[status] == event_value).cast(pl.Int8).to_numpy()
    X = drop_present(df, "rownames", time, status, *drop)
    return bunch_from(spec, X, event_time, indicator)


# ---- survival: breast cancer ----


def parse_rotterdam(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Rotterdam tumour bank, death endpoint (``dtime``/``death``).

    The cohort carries two endpoints. This is the overall-survival reading;
    ``rotterdam:recurrence`` and ``rotterdam:rfs`` are the other two. The
    unused endpoint's columns are dropped so they cannot leak.
    """
    return _simple(
        path, spec, time="dtime", status="death", drop=("pid", "rtime", "recur")
    )


def parse_rotterdam_recurrence(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Rotterdam, recurrence endpoint (``rtime``/``recur``)."""
    return _simple(
        path, spec, time="rtime", status="recur", drop=("pid", "dtime", "death")
    )


def parse_rotterdam_rfs(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Rotterdam, recurrence-free survival: the earlier of recurrence or death.

    This is the endpoint Royston & Altman use, and the one the DeepSurv
    ``gbsg`` benchmark inherits, so it is the reading to compare against
    published deep-survival numbers.
    """
    df = _read(path)
    rtime = df["rtime"].cast(pl.Float64).to_numpy()
    dtime = df["dtime"].cast(pl.Float64).to_numpy()
    recur = df["recur"].cast(pl.Int8).to_numpy()
    death = df["death"].cast(pl.Int8).to_numpy()
    event = ((recur == 1) | (death == 1)).astype(np.int8)
    # Event time is the first event; censored rows use the longer follow-up.
    first = np.where(recur == 1, rtime, dtime)
    event_time = np.where(
        event == 1, np.minimum(first, dtime), np.maximum(rtime, dtime)
    )
    X = drop_present(df, "rownames", "pid", "rtime", "recur", "dtime", "death")
    return bunch_from(spec, X, event_time, event)


# ---- survival: single-event cohorts ----


def parse_retinopathy(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Diabetic retinopathy laser trial: time to loss of vision.

    Two rows per patient (one eye treated, one untreated), so the rows are
    paired rather than independent; ``id`` is dropped but clustering remains.
    """
    return _simple(path, spec, time="futime", status="status", drop=("id",))


def parse_ovarian(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Edmunson ovarian cancer trial (n=26)."""
    return _simple(path, spec, time="futime", status="fustat")


def parse_stanford2(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Stanford heart transplant survival, post-transplant (n=184)."""
    return _simple(path, spec, time="time", status="status", drop=("id",))


def parse_myeloma(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Multiple myeloma survival (n=3882). ``entry`` is a left-truncation time."""
    return _simple(path, spec, time="futime", status="death", drop=("id",))


def parse_nafld(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Non-alcoholic fatty liver disease cohort (n=17549), death endpoint."""
    return _simple(path, spec, time="futime", status="status", drop=("id", "case.id"))


def parse_udca(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Ursodeoxycholic acid PBC trial (n=170)."""
    return _simple(path, spec, time="futime", status="status", drop=("id",))


def parse_aml(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Acute myelogenous leukemia maintenance trial (n=23).

    The smallest dataset in the registry, and the one every log-rank tutorial
    opens with. ``group`` is maintained vs non-maintained chemotherapy.
    """
    return _simple(path, spec, time="time", status="cens")


def parse_pharmaco_smoking(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Smoking-cessation trial: time to relapse (n=125)."""
    return _simple(path, spec, time="ttr", status="relapse", drop=("id",))


def parse_gastric_xelox(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Gastric cancer Xelox trial: progression-free survival in weeks (n=48)."""
    return _simple(path, spec, time="timeWeeks", status="delta")


def parse_ashkenazi(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """BRCA mutation carriers and breast cancer by age (n=3920).

    Age is the time scale, not follow-up duration. ``famID`` clusters
    relatives within families.
    """
    return _simple(path, spec, time="age", status="brcancer", drop=("famID",))


def parse_channing(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Channing House retirement home (n=462), the standard left-truncation example.

    ``ageentry`` is the age at entry and ``age`` the age at death or exit,
    both in months; ``time`` is the difference. The endpoint here is
    residence duration (``time``/``death``), with ``ageentry`` kept as a
    covariate so a caller can fit the left-truncated version instead.
    """
    return _simple(path, spec, time="time", status="death", drop=("obs", "age"))


# ---- survival: recurrent events, reduced to first event ----


def _first_event(
    path: Path,
    spec: DatasetSpec,
    *,
    enum: str,
    time: str,
    status: str,
    drop: tuple[str, ...],
) -> SurvivalBunch:
    """Keep each subject's first interval, turning a recurrent-event table
    into a right-censored time-to-first-event table."""
    df = _read(path).filter(pl.col(enum) == 1)
    event_time = df[time].cast(pl.Float64).to_numpy()
    indicator = df[status].cast(pl.Int8).to_numpy()
    X = drop_present(df, "rownames", enum, time, status, *drop)
    return bunch_from(spec, X, event_time, indicator)


def parse_bladder_first(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Bladder tumour recurrences, reduced to time to first recurrence (n=85).

    The source is a four-row-per-patient recurrent-event table; only the
    first interval is kept, so the result is an ordinary right-censored
    cohort rather than a counting-process one.
    """
    return _first_event(
        path, spec, enum="enum", time="stop", status="event", drop=("id",)
    )


def parse_cgd_first(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Chronic granulomatous disease, time to first serious infection (n=128)."""
    return _first_event(
        path,
        spec,
        enum="enum",
        time="tstop",
        status="status",
        drop=("id", "tstart", "random"),
    )


def parse_kidney_catheter(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Kidney dialysis catheter infections (n=76), the classic frailty example.

    Two records per patient. ``frail`` is the frailty estimate McGilchrist &
    Aisbett computed from this same data -- it is a fitted quantity, not a
    baseline covariate, and is dropped to keep the table honest.
    """
    return _simple(path, spec, time="time", status="status", drop=("id", "frail"))


def parse_valve_seat(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Diesel engine valve-seat replacements, time to first replacement (n=41).

    The source is a recurrent-event table over 41 engines; the first
    replacement per engine is kept and engines with none are censored at
    their last inspection.
    """
    df = _read(path)
    first = (
        df.sort("time")
        .group_by("id", maintain_order=True)
        .agg(pl.col("time").first(), pl.col("status").max())
    )
    event_time = first["time"].cast(pl.Float64).to_numpy()
    indicator = first["status"].cast(pl.Int8).to_numpy()
    # No baseline covariates survive the reduction: the engine id is not one.
    return bunch_from(spec, first.drop("id", "time", "status"), event_time, indicator)


# ---- survival: competing risks ----


def parse_transplant(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Liver transplant waiting list (n=815), three competing exits.

    ``event`` is one of ``censored`` / ``ltx`` (received a transplant) /
    ``withdraw`` / ``death``. Transplant is the outcome of interest, and
    death or withdrawal while waiting are the competing risks -- the
    textbook case where treating them as censoring overstates the
    transplant rate.
    """
    df = _read(path)
    event_time = df["futime"].cast(pl.Float64).to_numpy()
    labels = df["event"].to_numpy()
    cause = np.zeros(df.height, dtype=np.int8)
    cause[labels == "ltx"] = 1
    cause[labels == "withdraw"] = 2
    cause[labels == "death"] = 3
    X = drop_present(df, "rownames", "futime", "event")
    return bunch_from_competing(
        spec,
        X,
        event_time,
        cause,
        n_causes=3,
        cause_labels=("transplant", "withdrawal", "death on the list"),
    )


def parse_hoel(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Hoel mouse radiation experiment (n=181), three causes and no censoring.

    Every mouse died of exactly one of thymic lymphoma, reticulum cell
    sarcoma, or another cause, so the event rate is 100% and the whole
    question is *which* cause -- the cleanest competing-risks illustration
    in the registry.
    """
    df = _read(path)
    event_time = df["days"].cast(pl.Float64).to_numpy()
    outcome = df["outcome"].to_numpy()
    cause = np.zeros(df.height, dtype=np.int8)
    cause[outcome == "thymic lymphoma"] = 1
    cause[outcome == "reticulum cell sarcoma"] = 2
    cause[outcome == "other"] = 3
    X = drop_present(df, "rownames", "days", "outcome", "id")
    return bunch_from_competing(
        spec,
        X,
        event_time,
        cause,
        n_causes=3,
        cause_labels=("thymic lymphoma", "reticulum cell sarcoma", "other causes"),
    )


def parse_mgus_competing(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Original MGUS cohort (n=241): progression to malignancy vs death.

    Same first-event-wins recoding as ``mgus2``, on Kyle's earlier and
    smaller series. ``pcdx`` names the malignancy type and is dropped -- it
    is populated only for progressors and so leaks the cause.
    """
    df = _read(path)
    pctime = df["pctime"].cast(pl.Float64).to_numpy()
    futime = df["futime"].cast(pl.Float64).to_numpy()
    death = df["death"].cast(pl.Int8).to_numpy()
    progressed = np.isfinite(pctime) & ~df["pcdx"].is_null().to_numpy()

    pcm_first = progressed & ((death == 0) | (pctime <= futime))
    death_first = (death == 1) & ~pcm_first
    cause = np.zeros(df.height, dtype=np.int8)
    cause[pcm_first] = 1
    cause[death_first] = 2
    event_time = np.where(pcm_first, pctime, futime)
    X = drop_present(df, "rownames", "id", "pctime", "pcdx", "futime", "death")
    return bunch_from_competing(
        spec,
        X,
        event_time,
        cause,
        n_causes=2,
        cause_labels=("plasma cell malignancy", "death"),
    )


def parse_myeloid(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Acute myeloid leukemia trial (n=646), death endpoint.

    ``txtime`` / ``crtime`` / ``rltime`` record transplant, complete
    response and relapse -- intermediate states that occur *after*
    baseline, so they are dropped rather than offered as covariates.
    """
    return _simple(
        path,
        spec,
        time="futime",
        status="death",
        drop=("id", "txtime", "crtime", "rltime"),
    )


def parse_prostate_survival(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """SEER-derived prostate cancer cohort (n=14294), competing mortality.

    ``status``: ``0`` alive, ``1`` death from prostate cancer, ``2`` death
    from another cause. Most men in this cohort die of something else, which
    is exactly why it is the standard argument for cause-specific analysis.
    """
    df = _read(path)
    event_time = df["survTime"].cast(pl.Float64).to_numpy()
    cause = df["status"].cast(pl.Int8).to_numpy()
    X = drop_present(df, "rownames", "survTime", "status")
    return bunch_from_competing(
        spec,
        X,
        event_time,
        cause,
        n_causes=2,
        cause_labels=("prostate cancer death", "death from other causes"),
    )


def parse_hepato_cellular(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Hepatocellular carcinoma biomarker cohort (n=227), overall survival.

    48 columns of clinical and immunohistochemical markers. The
    recurrence endpoint (``RFS``/``Recurrence``) is the ``:rfs`` variant;
    whichever endpoint is not in use is dropped.
    """
    return _simple(
        path, spec, time="OS", status="Death", drop=("Number", "RFS", "Recurrence")
    )


def parse_hepato_cellular_rfs(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Hepatocellular carcinoma, recurrence-free survival endpoint."""
    return _simple(
        path, spec, time="RFS", status="Recurrence", drop=("Number", "OS", "Death")
    )


# ---- KMsurv: Klein & Moeschberger textbook cohorts ----


def parse_bmt(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Bone marrow transplant (n=137), disease-free survival.

    ``t2``/``d3`` is the disease-free survival endpoint: the earlier of
    relapse or death in remission. The competing-risks reading of the same
    rows is ``bmt:competing``. ``group`` is the disease-risk stratum (ALL,
    AML low risk, AML high risk); ``z1``-``z10`` are the covariates, and the
    other time columns are post-baseline and dropped.
    """
    df = _read(path)
    event_time = df["t2"].cast(pl.Float64).to_numpy()
    indicator = df["d3"].cast(pl.Int8).to_numpy()
    X = drop_present(
        df,
        "rownames",
        "t1",
        "t2",
        "d1",
        "d2",
        "d3",
        "ta",
        "da",
        "tc",
        "dc",
        "tp",
        "dp",
    )
    return bunch_from(spec, X, event_time, indicator)


def parse_bmt_competing(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Bone marrow transplant (n=137), relapse vs death in remission.

    The two ways a patient leaves remission are genuinely competing: a
    relapse cannot be observed after a treatment-related death. ``d2``
    marks relapse, and a disease-free-survival event (``d3``) without a
    relapse is a death in remission.
    """
    df = _read(path)
    event_time = df["t2"].cast(pl.Float64).to_numpy()
    relapse = df["d2"].cast(pl.Int8).to_numpy()
    dfs_event = df["d3"].cast(pl.Int8).to_numpy()
    cause = np.zeros(df.height, dtype=np.int8)
    cause[dfs_event == 1] = 2
    cause[relapse == 1] = 1
    X = drop_present(
        df,
        "rownames",
        "t1",
        "t2",
        "d1",
        "d2",
        "d3",
        "ta",
        "da",
        "tc",
        "dc",
        "tp",
        "dp",
    )
    return bunch_from_competing(
        spec,
        X,
        event_time,
        cause,
        n_causes=2,
        cause_labels=("relapse", "death in remission"),
    )


def parse_burn(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Burn wound infection trial (n=154): time to staphylococcus infection.

    ``Z1``-``Z11`` are the baseline covariates. ``T1``/``D1`` (excision) and
    ``T2``/``D2`` (prophylactic antibiotic) are post-baseline treatment
    events and are dropped.
    """
    return _simple(
        path,
        spec,
        time="T3",
        status="D3",
        drop=("Obs", "T1", "D1", "T2", "D2"),
    )


def parse_bfeed(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """National Survey of Family Growth breastfeeding duration (n=927)."""
    return _simple(path, spec, time="duration", status="delta")


def parse_std(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Sexually transmitted disease reinfection (n=877)."""
    return _simple(path, spec, time="time", status="rinfct", drop=("obs",))


def parse_pneumon(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Infant pneumonia by breastfeeding status (n=3470).

    ``chldage`` is age in months at hospitalisation for pneumonia or at
    censoring; ``hospital`` is the event. Only 73 events in 3470 infants,
    which makes this the most heavily censored cohort in the registry.
    """
    return _simple(path, spec, time="chldage", status="hospital")


def parse_hodg(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Hodgkin's lymphoma transplant survival (n=43)."""
    return _simple(path, spec, time="time", status="delta")


def parse_alloauto(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Allogeneic vs autologous transplant leukemia-free survival (n=101)."""
    return _simple(path, spec, time="time", status="delta")


def parse_btrial(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Breast cancer immunohistochemical response (n=45)."""
    return _simple(path, spec, time="time", status="death")


def parse_bnct(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Boron neutron capture therapy in rats (n=30)."""
    return _simple(path, spec, time="time", status="death")


def parse_drughiv(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """HIV drug trial, time to death (n=34)."""
    return _simple(path, spec, time="time", status="delta")


def parse_psych(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Iowa psychiatric inpatient survival (n=26), a left-truncation example."""
    return _simple(path, spec, time="time", status="death")


def parse_twins(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Danish twin survival after age 60 (n=24)."""
    return _simple(path, spec, time="age", status="death", drop=("id",))


def parse_kidney_infection(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Time to first exit-site kidney infection by catheter type (n=119).

    Klein & Moeschberger's Section 1.4 data. Distinct from R survival's
    ``kidney_catheter``, which is a different, recurrent-event study.
    """
    return _simple(path, spec, time="time", status="delta")


def parse_allograft(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Skin allograft survival by HLA match (n=34)."""
    return _simple(path, spec, time="time", status="rejection", drop=("patient",))


def parse_baboon(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Baboon descent times from sleeping trees (n=152).

    Time of day is the time scale and the "event" is being observed to
    descend; troops still in the tree when observation ended are censored.
    """
    return _simple(path, spec, time="time", status="observed", drop=("date",))


def parse_gehan(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Gehan's paired leukemia remission trial (n=42).

    21 matched pairs of patients, one given 6-mercaptopurine and one
    placebo, followed to relapse. The pairing is the design, so ``pair``
    is kept as a covariate rather than dropped -- it is the stratum a
    matched analysis conditions on.
    """
    return _simple(path, spec, time="time", status="cens")


def parse_mgus_death(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Original MGUS cohort, overall survival (``futime``/``death``).

    The plain single-event reading: 225 of 241 patients died, so only 6.6%
    are censored. Progression to malignancy is ignored here rather than
    competing -- for that, use ``mgus`` itself.
    """
    return _simple(
        path, spec, time="futime", status="death", drop=("id", "pctime", "pcdx")
    )
