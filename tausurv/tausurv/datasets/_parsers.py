"""Per-dataset parsers: cached raw file → :class:`SurvivalBunch`.

Each parser is a pure transformation from on-disk bytes to a Bunch. Metadata
(name, description, citation, license, url) travels in via the
:class:`DatasetSpec` so the registry is the single source of truth.

No imputation, no encoding, no scaling. The user gets the raw frame and
decides.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import polars as pl

from tausurv.datasets._bunch import SurvivalBunch
from tausurv.datasets._spec import DatasetSpec

_SUPPORT_COLUMNS = (
    "row_id",
    "age",
    "death",
    "sex",
    "hospdead",
    "slos",
    "d.time",
    "dzgroup",
    "dzclass",
    "num.co",
    "edu",
    "income",
    "scoma",
    "charges",
    "totcst",
    "totmcst",
    "avtisst",
    "race",
    "sps",
    "aps",
    "surv2m",
    "surv6m",
    "hday",
    "diabetes",
    "dementia",
    "ca",
    "prg2m",
    "prg6m",
    "dnr",
    "dnrday",
    "meanbp",
    "wblc",
    "hrt",
    "resp",
    "temp",
    "pafi",
    "alb",
    "bili",
    "crea",
    "sod",
    "ph",
    "glucose",
    "bun",
    "urine",
    "adlp",
    "adls",
    "sfdm2",
    "adlsc",
)


def parse_pbc(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Parse Mayo PBC. Recodes ``status`` so death is the single event.

    Source ``status``: ``0`` censored, ``1`` transplant, ``2`` death. Under
    a single-event analysis transplant is treated as censoring (the natural
    interpretation when the event of interest is death), matching the
    encoding used in Therneau & Grambsch's worked examples.
    """
    df = pl.read_csv(path)
    event_time = df["time"].cast(pl.Float64).to_numpy()
    event_indicator = (df["status"] == 2).cast(pl.Int8).to_numpy()
    X = df.drop("rownames", "id", "time", "status")
    return _bunch_from(spec, X, event_time, event_indicator)


def parse_rossi(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Parse the lifelines-hosted Rossi recidivism CSV.

    ``week`` is the time of first rearrest (or 52 if censored at one-year
    follow-up); ``arrest`` is the event indicator.
    """
    df = pl.read_csv(path)
    event_time = df["week"].cast(pl.Float64).to_numpy()
    event_indicator = df["arrest"].cast(pl.Int8).to_numpy()
    X = df.drop("week", "arrest")
    return _bunch_from(spec, X, event_time, event_indicator)


def parse_ifluid(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Parse the Nelson insulating-fluid voltage-stress test.

    All 41 specimens were run to breakdown — there is no status column,
    so ``event_indicator`` is all-ones. ``time`` is breakdown time in
    minutes; ``voltage`` (kV) is the stress covariate.
    """
    df = pl.read_csv(path)
    event_time = df["time"].cast(pl.Float64).to_numpy()
    event_indicator = np.ones(df.height, dtype=np.int8)
    X = df.drop("rownames", "time")
    return _bunch_from(spec, X, event_time, event_indicator)


def parse_imotor(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Parse the Nelson motor-insulation accelerated life test.

    40 motor specimens aged at four temperature levels (150-220°C). 17
    insulation failures observed, 23 right-censored at end of test.
    ``temp`` is the temperature covariate; ``time`` is hours to failure
    or censoring.
    """
    df = pl.read_csv(path)
    event_time = df["time"].cast(pl.Float64).to_numpy()
    event_indicator = df["status"].cast(pl.Int8).to_numpy()
    X = df.drop("rownames", "time", "status")
    return _bunch_from(spec, X, event_time, event_indicator)


def parse_capacitor(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Parse the capacitor accelerated life test.

    64 capacitors aged at combinations of temperature and voltage. The
    source has a ``fail`` column that records the rank within each
    stress group (a Type-II analysis artifact); it is dropped and the
    standard right-censored ``(time, status)`` is used. ``temperature``
    (°C) and ``voltage`` (V) are the ALT factors.
    """
    df = pl.read_csv(path)
    event_time = df["time"].cast(pl.Float64).to_numpy()
    event_indicator = df["status"].cast(pl.Int8).to_numpy()
    X = df.drop("rownames", "time", "status", "fail")
    return _bunch_from(spec, X, event_time, event_indicator)


def parse_genfan(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Parse the Meeker & Escobar diesel generator fan failure dataset.

    Tiny canonical reliability dataset: 70 fans observed for ``hours`` to
    failure; ``status`` is the failure indicator (12 failures, 58
    right-censored). ``hours`` is the only covariate channel besides the
    outcome, so X is empty — the parser keeps zero feature columns. This
    is the standard parametric-survival worked example (Weibull AFT) in
    Meeker & Escobar 1998.
    """
    df = pl.read_csv(path)
    event_time = df["hours"].cast(pl.Float64).to_numpy()
    event_indicator = df["status"].cast(pl.Int8).to_numpy()
    X = df.drop("rownames", "hours", "status")
    return _bunch_from(spec, X, event_time, event_indicator)


def parse_lung(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Parse the NCCTG lung-cancer cohort (R survival's ``cancer``).

    Source ``status``: ``1`` censored, ``2`` dead. Recoded to ``0/1`` so
    that ``1`` is the event. Several covariates carry real-world
    missingness (``ph.ecog``, ``ph.karno``, ``pat.karno``, ``meal.cal``,
    ``wt.loss``); users decide how to impute or drop.
    """
    df = pl.read_csv(path)
    event_time = df["time"].cast(pl.Float64).to_numpy()
    event_indicator = (df["status"] == 2).cast(pl.Int8).to_numpy()
    X = df.drop("rownames", "time", "status")
    return _bunch_from(spec, X, event_time, event_indicator)


def parse_veteran(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Parse the VA lung-cancer cohort.

    ``status`` is already ``0/1`` (event = death). ``celltype`` is a
    string-typed factor with levels squamous / smallcell / adeno / large.
    """
    df = pl.read_csv(path)
    event_time = df["time"].cast(pl.Float64).to_numpy()
    event_indicator = df["status"].cast(pl.Int8).to_numpy()
    X = df.drop("rownames", "time", "status")
    return _bunch_from(spec, X, event_time, event_indicator)


def parse_flchain(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Parse the Mayo Clinic free-light-chain population cohort.

    ``futime`` is follow-up in days; ``death`` is the event indicator.
    The ``chapter`` column (cause-of-death category, only populated for
    decedents) is dropped because it perfectly leaks the outcome under a
    single-event analysis. A future competing-risks loader will surface it
    as ``cause``.
    """
    df = pl.read_csv(path)
    event_time = df["futime"].cast(pl.Float64).to_numpy()
    event_indicator = df["death"].cast(pl.Int8).to_numpy()
    X = df.drop("rownames", "futime", "death", "chapter")
    return _bunch_from(spec, X, event_time, event_indicator)


def parse_gbsg(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Parse the GBSG breast-cancer cohort.

    ``rfstime`` is recurrence-free survival in days; ``status`` is ``1`` for
    recurrence or death and ``0`` for censoring. ``pid`` is the subject
    identifier and is dropped.
    """
    df = pl.read_csv(path)
    event_time = df["rfstime"].cast(pl.Float64).to_numpy()
    event_indicator = df["status"].cast(pl.Int8).to_numpy()
    X = df.drop("rownames", "pid", "rfstime", "status")
    return _bunch_from(spec, X, event_time, event_indicator)


def parse_melanoma(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Parse the Andersen melanoma competing-risks cohort.

    Source ``status`` is the original 3-level coding: ``1`` died from
    melanoma, ``2`` alive at end of follow-up (censored), ``3`` died from
    other causes. We remap to standard competing-risks form:

    - ``cause = 1`` (melanoma death)
    - ``cause = 2`` (death from other causes)
    - ``cause = 0`` (censored)
    """
    df = pl.read_csv(path)
    event_time = df["time"].cast(pl.Float64).to_numpy()
    status = df["status"].to_numpy()
    cause = np.zeros(df.height, dtype=np.int8)
    cause[status == 1] = 1
    cause[status == 3] = 2
    X = df.drop("rownames", "time", "status")
    return _bunch_from_competing(
        spec,
        X,
        event_time,
        cause,
        n_causes=2,
        cause_labels=("melanoma death", "death from other causes"),
    )


def parse_mgus2(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Parse the Mayo MGUS-to-malignancy competing-risks cohort.

    The R survival source ships two parallel ``(time, indicator)`` pairs:
    ``(ptime, pstat)`` records progression to plasma-cell malignancy and
    ``(futime, death)`` records death. We recode to first-event-wins
    competing-risks form:

    - ``cause = 1`` (PCM) if ``pstat == 1`` and PCM happens before death;
    - ``cause = 2`` (death) if ``death == 1`` and death happens first;
    - ``cause = 0`` (censored) otherwise.

    ``event_time`` is the time of that first event (or last contact if
    censored). After the recoding ``ptime``/``pstat``/``futime``/``death``
    are dropped from ``X``; the survivor covariates are age, sex, dxyr,
    hgb, creat, mspike.
    """
    df = pl.read_csv(path)
    ptime = df["ptime"].to_numpy().astype(np.float64)
    pstat = df["pstat"].to_numpy().astype(np.int8)
    futime = df["futime"].to_numpy().astype(np.float64)
    death = df["death"].to_numpy().astype(np.int8)

    pcm_first = (pstat == 1) & ((death == 0) | (ptime <= futime))
    death_first = (death == 1) & ~pcm_first

    cause = np.zeros(df.height, dtype=np.int8)
    cause[pcm_first] = 1
    cause[death_first] = 2
    event_time = np.where(pcm_first, ptime, futime)

    X = df.drop("rownames", "id", "ptime", "pstat", "futime", "death")
    return _bunch_from_competing(
        spec,
        X,
        event_time,
        cause,
        n_causes=2,
        cause_labels=("plasma cell malignancy", "death"),
    )


def parse_support(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Parse the SUPPORT critical-care cohort from the hbiostat zip.

    Two quirks the parser smooths over:

    The hbiostat CSV header lists 47 column names but each data row has 48
    fields — the first is an unnamed integer row index. We supply ``row_id``
    as the missing leading name and drop it after the read.

    Most numeric columns are stored with leading whitespace
    (``" 62.84998"``), which makes polars infer them as strings. Each
    string column is whitespace-stripped and cast to ``Float64`` when the
    cast succeeds; genuine categorical columns (``sex``, ``dzgroup``,
    ``dzclass``, ``income``, ``race``, ``ca``, ``dnr``, ``sfdm2``) stay
    string-typed.
    """
    with zipfile.ZipFile(path) as zf, zf.open("support2.csv") as f:
        df = pl.read_csv(
            f.read(),
            has_header=False,
            skip_rows=1,
            null_values=["NA", ""],
            new_columns=list(_SUPPORT_COLUMNS),
            infer_schema_length=10000,
        )
    df = _coerce_numeric_strings(df)
    event_time = df["d.time"].cast(pl.Float64).to_numpy()
    event_indicator = df["death"].cast(pl.Int8).to_numpy()
    X = df.drop("row_id", "d.time", "death")
    return _bunch_from(spec, X, event_time, event_indicator)


def _coerce_numeric_strings(df: pl.DataFrame) -> pl.DataFrame:
    """Strip whitespace from string columns; cast to ``Float64`` where valid.

    The SUPPORT CSV pads numeric values with leading spaces, which makes
    polars infer them as strings. This recovers the numeric dtype while
    leaving genuine categorical columns as strings (their cast yields
    almost-all nulls, which is the discriminator).
    """
    for col, dtype in zip(df.columns, df.dtypes, strict=True):
        if dtype != pl.String:
            continue
        stripped = df[col].str.strip_chars()
        casted = stripped.cast(pl.Float64, strict=False)
        if casted.is_not_null().sum() >= stripped.is_not_null().sum() * 0.5:
            df = df.with_columns(casted.alias(col))
        else:
            df = df.with_columns(stripped.alias(col))
    return df


def parse_nwtco(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Parse the National Wilms Tumor case-cohort cohort.

    ``rel`` is the relapse indicator; ``edrel`` is days from diagnosis to
    relapse or last follow-up. The full cohort is 4028 subjects with the
    ``in.subcohort`` flag marking the case-cohort subsample, kept as a
    covariate so users can run case-cohort or full-cohort analyses.
    """
    df = pl.read_csv(path)
    event_time = df["edrel"].cast(pl.Float64).to_numpy()
    event_indicator = df["rel"].cast(pl.Int8).to_numpy()
    X = df.drop("rownames", "seqno", "rel", "edrel")
    return _bunch_from(spec, X, event_time, event_indicator)


def parse_colon(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Parse the Moertel colon-cancer adjuvant-therapy trial as competing risks.

    The R survival source stores two rows per subject — one for recurrence
    (``etype=1``) and one for death (``etype=2``) — with parallel
    ``(time, status)`` pairs. The loader pivots to wide form and recodes
    to first-event-wins: cause 1 (recurrence), cause 2 (death) when death
    occurs before any recurrence, cause 0 (censored) otherwise.
    """
    df = pl.read_csv(path)
    rec = df.filter(pl.col("etype") == 1).sort("id")
    dth = df.filter(pl.col("etype") == 2).sort("id")
    r_time = rec["time"].to_numpy().astype(np.float64)
    r_stat = rec["status"].to_numpy()
    d_time = dth["time"].to_numpy().astype(np.float64)
    d_stat = dth["status"].to_numpy()
    rec_first = (r_stat == 1) & ((d_stat == 0) | (r_time <= d_time))
    death_first = (d_stat == 1) & ~rec_first
    cause = np.zeros(rec.height, dtype=np.int8)
    cause[rec_first] = 1
    cause[death_first] = 2
    event_time = np.where(rec_first, r_time, d_time)
    X = rec.drop("rownames", "id", "time", "status", "etype")
    return _bunch_from_competing(
        spec,
        X,
        event_time,
        cause,
        n_causes=2,
        cause_labels=("recurrence", "death"),
    )


def parse_larynx(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Parse the Kosary laryngeal-cancer cohort (Klein-Moeschberger Ch 1).

    90 male patients with cancer of the larynx, stratified by disease
    stage I-IV. ``delta`` is the death indicator; ``time`` is in months.
    """
    df = pl.read_csv(path)
    event_time = df["time"].cast(pl.Float64).to_numpy()
    event_indicator = df["delta"].cast(pl.Int8).to_numpy()
    X = df.drop("rownames", "time", "delta")
    return _bunch_from(spec, X, event_time, event_indicator)


def parse_tongue(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Parse the Sickle-Schmidt tongue-cancer cohort (Klein-Moeschberger Ch 1).

    80 patients with squamous-cell carcinoma of the tongue, stratified
    into aneuploid (``type=1``) and diploid (``type=2``) tumour DNA
    profiles. ``delta`` is the death indicator; ``time`` is in weeks.
    """
    df = pl.read_csv(path)
    event_time = df["time"].cast(pl.Float64).to_numpy()
    event_indicator = df["delta"].cast(pl.Int8).to_numpy()
    X = df.drop("rownames", "time", "delta")
    return _bunch_from(spec, X, event_time, event_indicator)


def parse_kidney_transplant(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Parse the UNOS kidney-transplant cohort (lifelines mirror).

    863 transplant recipients followed for death. Race-by-sex appears
    pre-one-hot-encoded in the source as ``black_male``, ``white_male``,
    ``black_female`` (with ``white_female`` as the reference category);
    they are kept as-is.
    """
    df = pl.read_csv(path)
    event_time = df["time"].cast(pl.Float64).to_numpy()
    event_indicator = df["death"].cast(pl.Int8).to_numpy()
    X = df.drop("time", "death")
    return _bunch_from(spec, X, event_time, event_indicator)


def parse_waltons(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Parse the lifelines miR-137 KM teaching demo.

    163 subjects split into ``miR-137`` and ``control`` groups; the
    canonical two-sample KM demonstration in lifelines documentation.
    """
    df = pl.read_csv(path)
    event_time = df["T"].cast(pl.Float64).to_numpy()
    event_indicator = df["E"].cast(pl.Int8).to_numpy()
    X = df.drop("T", "E")
    return _bunch_from(spec, X, event_time, event_indicator)


def parse_telco_churn(path: Path, spec: DatasetSpec) -> SurvivalBunch:
    """Parse the IBM Watson Telco customer-churn sample.

    7043 telecom subscribers; ``tenure`` is months as customer (event
    time), ``Churn`` is the "Yes"/"No" event indicator recoded to 0/1.
    Many string-typed service-plan covariates are preserved untouched.
    ``TotalCharges`` is whitespace for zero-tenure customers; treated as
    null on read.
    """
    df = pl.read_csv(path, null_values=[" ", ""])
    event_time = df["tenure"].cast(pl.Float64).to_numpy()
    event_indicator = (df["Churn"] == "Yes").cast(pl.Int8).to_numpy()
    X = df.drop("customerID", "tenure", "Churn")
    return _bunch_from(spec, X, event_time, event_indicator)


def _bunch_from(
    spec: DatasetSpec,
    X: pl.DataFrame,
    event_time: np.ndarray,
    event_indicator: np.ndarray,
) -> SurvivalBunch:
    return SurvivalBunch(
        X=X,
        event_time=event_time,
        event_indicator=event_indicator,
        feature_names=tuple(X.columns),
        name=spec.name,
        description=spec.description,
        citation=spec.citation,
        license=spec.license,
        url=spec.url or "",
        time_unit=spec.time_unit,
        tags=spec.tags,
    )


def _bunch_from_competing(
    spec: DatasetSpec,
    X: pl.DataFrame,
    event_time: np.ndarray,
    cause: np.ndarray,
    *,
    n_causes: int,
    cause_labels: tuple[str, ...] | None,
) -> SurvivalBunch:
    cause = cause.astype(np.int8)
    return SurvivalBunch(
        X=X,
        event_time=event_time,
        event_indicator=(cause > 0).astype(np.int8),
        feature_names=tuple(X.columns),
        name=spec.name,
        description=spec.description,
        citation=spec.citation,
        license=spec.license,
        url=spec.url or "",
        cause=cause,
        n_causes=n_causes,
        cause_labels=cause_labels,
        time_unit=spec.time_unit,
        tags=spec.tags,
    )
