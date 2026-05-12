r"""Survival dataset loaders.

Generic API
-----------

.. code-block:: python

    from tausurv.datasets import load_dataset, list_datasets, dataset_info

    list_datasets()                    # → ['gbsg', 'pbc', 'rossi']
    info = dataset_info("pbc")         # metadata only, no download
    ds   = load_dataset("pbc")         # SurvivalBunch; downloads on first
                                       # use, hits the cache afterwards

Per-dataset shortcuts (autocomplete-friendly):

.. code-block:: python

    from tausurv.datasets import load_pbc
    ds = load_pbc()

The two routes are equivalent; both auto-fetch on first use and read from
the cache thereafter.

Cache resolution priority: ``cache_dir`` kwarg, ``TAUSURV_DATA`` env var,
then ``platformdirs.user_cache_dir("tausurv")/datasets``. Set
``TAUSURV_OFFLINE=1`` to make any would-be fetch raise instead.
"""

from __future__ import annotations

from pathlib import Path

from tausurv.datasets._bunch import SurvivalBunch
from tausurv.datasets._cache import resolve_cache_dir
from tausurv.datasets._errors import (
    CredentialedDatasetError,
    DatasetIntegrityError,
    OfflineModeError,
    UnknownDatasetError,
    UserProvidedDatasetError,
)
from tausurv.datasets._registry import (
    dataset_info,
    list_datasets,
    load_dataset,
)
from tausurv.datasets._spec import Access, DatasetInfo


def load_pbc(
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> SurvivalBunch:
    """Mayo Clinic primary biliary cholangitis trial (n=418).

    See :func:`dataset_info` for the full description and citation.
    """
    return load_dataset("pbc", cache_dir=cache_dir, force_download=force_download)


def load_rossi(
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> SurvivalBunch:
    """Rossi prison-release recidivism cohort (n=432).

    See :func:`dataset_info` for the full description and citation.
    """
    return load_dataset("rossi", cache_dir=cache_dir, force_download=force_download)


def load_gbsg(
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> SurvivalBunch:
    """German Breast Cancer Study Group cohort (n=686).

    See :func:`dataset_info` for the full description and citation.
    """
    return load_dataset("gbsg", cache_dir=cache_dir, force_download=force_download)


def load_genfan(
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> SurvivalBunch:
    """Meeker & Escobar generator fan failure data (n=70, hours).

    The textbook Weibull-AFT reliability example. See :func:`dataset_info`
    for the full description and citation.
    """
    return load_dataset(
        "genfan", cache_dir=cache_dir, force_download=force_download
    )


def load_ifluid(
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> SurvivalBunch:
    """Nelson insulating-fluid voltage-stress test (n=41, minutes).

    All-event Weibull-AFT example with voltage as the stress covariate.
    See :func:`dataset_info` for the full description and citation.
    """
    return load_dataset(
        "ifluid", cache_dir=cache_dir, force_download=force_download
    )


def load_imotor(
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> SurvivalBunch:
    """Nelson motor-insulation accelerated life test (n=40, hours).

    Temperature-stressed ALT example; 17 failures, 23 censored. See
    :func:`dataset_info` for the full description and citation.
    """
    return load_dataset(
        "imotor", cache_dir=cache_dir, force_download=force_download
    )


def load_capacitor(
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> SurvivalBunch:
    """Capacitor accelerated life test (n=64, hours).

    Two-factor ALT (temperature x voltage), 32 failures. See
    :func:`dataset_info` for the full description and citation.
    """
    return load_dataset(
        "capacitor", cache_dir=cache_dir, force_download=force_download
    )


def load_lung(
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> SurvivalBunch:
    """NCCTG advanced lung-cancer cohort (n=228).

    See :func:`dataset_info` for the full description and citation.
    """
    return load_dataset("lung", cache_dir=cache_dir, force_download=force_download)


def load_veteran(
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> SurvivalBunch:
    """Veterans Administration lung-cancer trial (n=137).

    See :func:`dataset_info` for the full description and citation.
    """
    return load_dataset(
        "veteran", cache_dir=cache_dir, force_download=force_download
    )


def load_flchain(
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> SurvivalBunch:
    """Mayo free-light-chain population cohort (n=7874).

    See :func:`dataset_info` for the full description and citation.
    """
    return load_dataset(
        "flchain", cache_dir=cache_dir, force_download=force_download
    )


def load_melanoma(
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> SurvivalBunch:
    """Andersen melanoma competing-risks cohort (n=205, K=2).

    Two competing causes: death from melanoma (cause 1) and death from
    other causes (cause 2). See :func:`dataset_info` for the full
    description and citation.
    """
    return load_dataset(
        "melanoma", cache_dir=cache_dir, force_download=force_download
    )


def load_mgus2(
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> SurvivalBunch:
    """Mayo MGUS-to-malignancy competing-risks cohort (n=1384, K=2).

    Two competing causes: plasma-cell malignancy (cause 1) and death from
    other causes (cause 2). The returned bunch carries ``cause``,
    ``n_causes=2``, and ``cause_labels=("plasma cell malignancy", "death")``.
    See :func:`dataset_info` for the full description and citation.
    """
    return load_dataset(
        "mgus2", cache_dir=cache_dir, force_download=force_download
    )


def load_support(
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> SurvivalBunch:
    """SUPPORT seriously-ill hospitalised adults cohort (n=9105).

    See :func:`dataset_info` for the full description and citation,
    including the list of outcome-leaking columns to drop for a clean
    baseline-prediction task.
    """
    return load_dataset(
        "support", cache_dir=cache_dir, force_download=force_download
    )


def load_colon(
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> SurvivalBunch:
    """Moertel colon-cancer trial as competing risks (n=929, K=2).

    Cause 1 = recurrence, cause 2 = death without prior recurrence.
    See :func:`dataset_info` for the full description and citation.
    """
    return load_dataset(
        "colon", cache_dir=cache_dir, force_download=force_download
    )


def load_kidney_transplant(
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> SurvivalBunch:
    """UNOS kidney transplant cohort (n=863).

    See :func:`dataset_info` for the full description and citation.
    """
    return load_dataset(
        "kidney_transplant",
        cache_dir=cache_dir,
        force_download=force_download,
    )


def load_larynx(
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> SurvivalBunch:
    """Kosary laryngeal-cancer cohort (n=90, months).

    Klein-Moeschberger Chapter 1 stage-stratified example. See
    :func:`dataset_info` for the full description and citation.
    """
    return load_dataset(
        "larynx", cache_dir=cache_dir, force_download=force_download
    )


def load_nwtco(
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> SurvivalBunch:
    """National Wilms Tumor Study cohort (n=4028).

    Two-phase case-cohort design; the ``in.subcohort`` flag is kept as a
    covariate so users can fit either case-cohort or full-cohort models.
    See :func:`dataset_info` for the full description and citation.
    """
    return load_dataset(
        "nwtco", cache_dir=cache_dir, force_download=force_download
    )


def load_telco_churn(
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> SurvivalBunch:
    """IBM Watson Telco customer-churn sample (n=7043, months).

    Industry-standard churn-survival benchmark. See :func:`dataset_info`
    for the full description and citation.
    """
    return load_dataset(
        "telco_churn", cache_dir=cache_dir, force_download=force_download
    )


def load_tongue(
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> SurvivalBunch:
    """Sickle-Santanello tongue-cancer cohort (n=80, weeks).

    Klein-Moeschberger Chapter 1 two-sample KM example. See
    :func:`dataset_info` for the full description and citation.
    """
    return load_dataset(
        "tongue", cache_dir=cache_dir, force_download=force_download
    )


def load_waltons(
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> SurvivalBunch:
    """lifelines miR-137 vs control KM demo (n=163).

    See :func:`dataset_info` for the full description and citation.
    """
    return load_dataset(
        "waltons", cache_dir=cache_dir, force_download=force_download
    )


__all__ = [
    "Access",
    "CredentialedDatasetError",
    "DatasetInfo",
    "DatasetIntegrityError",
    "OfflineModeError",
    "SurvivalBunch",
    "UnknownDatasetError",
    "UserProvidedDatasetError",
    "dataset_info",
    "list_datasets",
    "load_capacitor",
    "load_colon",
    "load_dataset",
    "load_flchain",
    "load_gbsg",
    "load_genfan",
    "load_ifluid",
    "load_imotor",
    "load_kidney_transplant",
    "load_larynx",
    "load_lung",
    "load_melanoma",
    "load_mgus2",
    "load_nwtco",
    "load_pbc",
    "load_rossi",
    "load_support",
    "load_telco_churn",
    "load_tongue",
    "load_veteran",
    "load_waltons",
    "resolve_cache_dir",
]
