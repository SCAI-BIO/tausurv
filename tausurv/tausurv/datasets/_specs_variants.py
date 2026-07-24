"""Variant entries for cohorts whose base spec lives in the core registry.

These are built from their base spec rather than declared beside it, so a
variant cannot drift onto a different source file than the study it claims to
be a reading of: ``url`` and ``sha256`` are inherited, never retyped.
"""

from __future__ import annotations

from tausurv.datasets import _parsers as p
from tausurv.datasets._spec import DatasetSpec, Parser


def _variant(
    base: DatasetSpec,
    suffix: str,
    *,
    parser: Parser,
    description: str,
    tags: tuple[str, ...] | None = None,
    time_unit: str | None = None,
    citation: str | None = None,
) -> DatasetSpec:
    """A named reading of ``base``, sharing its source file and licence."""
    return DatasetSpec(
        name=f"{base.name}:{suffix}",
        access=base.access,
        url=base.url,
        sha256=base.sha256,
        license=base.license,
        citation=citation or base.citation,
        description=description,
        parser=parser,
        tags=tags if tags is not None else base.tags,
        time_unit=time_unit or base.time_unit,
    )


def build(registry: dict[str, DatasetSpec]) -> dict[str, DatasetSpec]:
    """Construct every variant whose base is already in ``registry``."""
    pbc, lung = registry["pbc"], registry["lung"]
    flchain, support, colon = registry["flchain"], registry["support"], registry["colon"]

    return {
        "pbc:randomised": _variant(
            pbc,
            "randomised",
            parser=p.parse_pbc_randomised,
            description=(
                "The 312 randomised patients from the Mayo PBC trial, 125 "
                "deaths. Therneau and Grambsch's Cox examples use this "
                "subset rather than all 418: the other 106 patients "
                "consented to follow-up but not to randomisation, and are "
                "missing most lab covariates. Reported PBC results are "
                "split roughly evenly between the two tables, so the row "
                "count is worth checking before comparing numbers."
            ),
            tags=("clinical", "trial"),
        ),
        "pbc:transplant": _variant(
            pbc,
            "transplant",
            parser=p.parse_pbc_transplant,
            description=(
                "Mayo PBC as competing risks: death (161) versus liver "
                "transplant (25), across all 418 patients. The base `pbc` "
                "entry censors transplants, which is correct when the "
                "question is about death and wrong when it is about disease "
                "course -- a transplant removes a patient from risk "
                "precisely because their disease progressed."
            ),
            tags=("clinical", "competing-risks"),
        ),
        "flchain:complete": _variant(
            flchain,
            "complete",
            parser=p.parse_flchain_complete,
            description=(
                "The 6524 FLCHAIN subjects with a complete covariate "
                "record, 1962 deaths (30.1%). Dropping the 1350 subjects "
                "missing a creatinine assay is the processing used by the "
                "deep-survival benchmark tables. It is not innocuous: the "
                "event rate moves from 27.5% to 30.1%, so missingness here "
                "is informative and the two tables are not interchangeable."
            ),
        ),
        "support:nonleaky": _variant(
            support,
            "nonleaky",
            parser=p.parse_support_nonleaky,
            description=(
                "SUPPORT with the outcome-derived columns removed: 9105 "
                "patients, 31 baseline covariates. The raw cohort ships the "
                "SUPPORT prognostic model's own survival estimates "
                "(`surv2m`, `surv6m`), the physician's estimates (`prg2m`, "
                "`prg6m`), in-hospital death, length of stay and accrued "
                "costs. `surv6m` alone reaches c-index 0.72 against the "
                "endpoint, above published models fitted on real "
                "covariates, so any result on the raw table is an artefact "
                "of reading the answer off a column. Use this variant for "
                "benchmarking; the base entry stays raw so the leak stays "
                "inspectable."
            ),
            tags=("clinical", "benchmark"),
        ),
        "colon:recurrence": _variant(
            colon,
            "recurrence",
            parser=p.parse_colon_recurrence,
            description=(
                "The Moertel colon cancer trial with recurrence as a single "
                "endpoint (468 recurrences in 929 patients), death treated "
                "as censoring. This is the reading used in most Cox "
                "demonstrations of the trial; `colon` itself is the "
                "competing-risks version."
            ),
            tags=("clinical", "trial"),
        ),
        "colon:death": _variant(
            colon,
            "death",
            parser=p.parse_colon_death,
            description=(
                "The Moertel colon cancer trial with overall survival as a "
                "single endpoint (452 deaths in 929 patients). Together "
                "with `colon:recurrence` this decomposes the competing-risks "
                "`colon` entry into the two marginal analyses."
            ),
            tags=("clinical", "trial"),
        ),
        "lung:complete": _variant(
            lung,
            "complete",
            parser=p.parse_lung_complete,
            description=(
                "The 168 NCCTG lung cancer patients with no missing "
                "covariates, 121 deaths. Roughly a quarter of the cohort is "
                "dropped, which is why methods that handle missingness "
                "natively should be compared on `lung` and not here. The "
                "enrolling institution is dropped before the complete-case "
                "filter, since it is an administrative code rather than a "
                "predictor."
            ),
        ),
    }
