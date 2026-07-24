"""Dataset registry and the generic loaders.

The registry is the single source of truth: every per-dataset shortcut in
``tausurv.datasets.__init__`` resolves to a :class:`DatasetSpec` from here.

A name is ``base`` or ``base:variant``. The base name always resolves to the
reading a survival textbook would call default; a variant names one other
published processing of the same study. Aliases cover the names the same
cohort travels under in different literatures -- ``gbsg2`` and ``actg320``
resolve, but do not appear in :func:`list_datasets`, so there is exactly one
canonical name per table.

Entries are declared across four modules: the core cohorts here, the
R-ecosystem cohorts in ``_specs_r``, the benchmark and gated cohorts in
``_specs_bench``, and variants of the core cohorts in ``_specs_variants``.
"""

from __future__ import annotations

from pathlib import Path

from tausurv.datasets._bunch import SurvivalBunch
from tausurv.datasets._cache import fetch_to_cache
from tausurv.datasets._errors import (
    CredentialedDatasetError,
    MissingDependencyError,
    UnknownDatasetError,
    UserProvidedDatasetError,
)
from tausurv.datasets._parsers import (
    parse_capacitor,
    parse_colon,
    parse_flchain,
    parse_gbsg,
    parse_genfan,
    parse_ifluid,
    parse_imotor,
    parse_kidney_transplant,
    parse_larynx,
    parse_lung,
    parse_melanoma,
    parse_mgus2,
    parse_nwtco,
    parse_pbc,
    parse_rossi,
    parse_support,
    parse_telco_churn,
    parse_tongue,
    parse_veteran,
    parse_waltons,
)
from tausurv.datasets._spec import Access, DatasetInfo, DatasetSpec, split_name

_REGISTRY: dict[str, DatasetSpec] = {
    "pbc": DatasetSpec(
        name="pbc",
        access=Access.OPEN,
        url="https://vincentarelbundock.github.io/Rdatasets/csv/survival/pbc.csv",
        sha256="797ea9b6abfec34297ef07f361a2e0bfdd90c3c2def180adf8547ea30e75b613",
        license="LGPL-2.1-or-later (R survival via Rdatasets)",
        citation=(
            "Therneau, T. & Grambsch, P. (2000). Modeling Survival Data: "
            "Extending the Cox Model. Springer."
        ),
        description=(
            "Mayo Clinic primary biliary cholangitis trial (1974-1984), 418 "
            "patients. Status is recoded for single-event analysis: death = "
            "event (1), transplant and end-of-study = censored (0). The "
            "first 312 patients were randomised to D-penicillamine or "
            "placebo; the remaining 106 are non-randomised follow-up. "
            "Real-world missingness in several lab covariates."
        ),
        parser=parse_pbc,
        tags=("clinical",),
        time_unit="days",
    ),
    "rossi": DatasetSpec(
        name="rossi",
        access=Access.OPEN,
        url=(
            "https://raw.githubusercontent.com/CamDavidsonPilon/lifelines/"
            "master/lifelines/datasets/rossi.csv"
        ),
        sha256="0214400170e07f3015a285edca2014cac0dae8aac125fcbaad642490e05f892c",
        license="MIT (via lifelines)",
        citation=(
            "Rossi, P. H., Berk, R. A., & Lenihan, K. J. (1980). Money, "
            "Work and Crime: Experimental Evidence. Academic Press."
        ),
        description=(
            "Recidivism cohort of 432 prison releasees followed for one "
            "year. Outcome is the week of first rearrest (event) or "
            "one-year follow-up (censoring). Covariates: financial aid "
            "(randomised), age at release, race, work experience, marital "
            "status, parole status, prior convictions. Classic textbook "
            "Cox example."
        ),
        parser=parse_rossi,
        tags=("recidivism",),
        time_unit="weeks",
    ),
    "capacitor": DatasetSpec(
        name="capacitor",
        access=Access.OPEN,
        url="https://vincentarelbundock.github.io/Rdatasets/csv/survival/capacitor.csv",
        sha256="4b57443c4ff8855fcc6c746f6f2c003c1ea2790dff898e75d7bfae9546141da4",
        license="LGPL-2.1-or-later (R survival via Rdatasets)",
        citation=(
            "Meeker, W. Q. & Escobar, L. A. (1998). Statistical Methods "
            "for Reliability Data. Wiley. Capacitor accelerated life test "
            "example."
        ),
        description=(
            "Capacitor accelerated life test: 64 units aged at "
            "combinations of temperature (170-200°C) and voltage "
            "(200-300 V). 32 failures, 32 right-censored. The source "
            "carries a ``fail`` column (rank within each stress group, "
            "used for Type-II analysis) which the loader drops. ALT "
            "factors are temperature and voltage."
        ),
        parser=parse_capacitor,
        tags=("reliability",),
        time_unit="hours",
    ),
    "ifluid": DatasetSpec(
        name="ifluid",
        access=Access.OPEN,
        url="https://vincentarelbundock.github.io/Rdatasets/csv/survival/ifluid.csv",
        sha256="e52016004044ccf4de91906e0c0effbb01389739ca1a2175f0686d49c66cfefe",
        license="LGPL-2.1-or-later (R survival via Rdatasets)",
        citation=(
            "Nelson, W. B. (1990). Accelerated Testing: Statistical "
            "Models, Test Plans, and Data Analyses. Wiley. Insulating "
            "fluid breakdown experiment."
        ),
        description=(
            "Breakdown times of an insulating fluid under voltage stress "
            "(Nelson 1972/1990): 41 specimens at seven voltage levels "
            "(26-38 kV), all run to breakdown. No censoring — the "
            "event_indicator is all-ones. Cleanest Weibull-AFT-with-"
            "covariate example in the small-reliability set; the "
            "log-linear voltage effect is the textbook Inverse Power Law."
        ),
        parser=parse_ifluid,
        tags=("reliability",),
        time_unit="minutes",
    ),
    "imotor": DatasetSpec(
        name="imotor",
        access=Access.OPEN,
        url="https://vincentarelbundock.github.io/Rdatasets/csv/survival/imotor.csv",
        sha256="9028f967885eec7e4626d70de2b2bbaae9175ab798387264bd984071a3ffb045",
        license="LGPL-2.1-or-later (R survival via Rdatasets)",
        citation=(
            "Nelson, W. B. (1990). Accelerated Testing: Statistical "
            "Models, Test Plans, and Data Analyses. Wiley. Motor "
            "insulation aging study."
        ),
        description=(
            "Motor insulation accelerated life test: 40 specimens aged "
            "at four temperatures (150, 170, 190, 220°C). 17 insulation "
            "failures, 23 right-censored at end of test. ``temp`` is the "
            "Arrhenius stress covariate; the textbook example for "
            "temperature-driven ALT with the lognormal AFT."
        ),
        parser=parse_imotor,
        tags=("reliability",),
        time_unit="hours",
    ),
    "genfan": DatasetSpec(
        name="genfan",
        access=Access.OPEN,
        url="https://vincentarelbundock.github.io/Rdatasets/csv/survival/genfan.csv",
        sha256="6e51ce0cf04c175c457c897ba537c2fd3b40923c870fde7762555ddcb3d4ea3a",
        license="LGPL-2.1-or-later (R survival via Rdatasets)",
        citation=(
            "Meeker, W. Q. & Escobar, L. A. (1998). Statistical Methods "
            "for Reliability Data. Wiley. Example 8.4 (diesel generator "
            "fan failures)."
        ),
        description=(
            "Diesel generator fan failure-time data, 70 units observed "
            "for failure or right-censoring at end of test. 12 failures, "
            "58 censored. The textbook introductory Weibull-AFT teaching "
            "example in reliability engineering; the dataset has no "
            "covariates besides the outcome, so this is a single-sample "
            "lifetime fit. Time is in operating hours."
        ),
        parser=parse_genfan,
        tags=("reliability",),
        time_unit="hours",
    ),
    "lung": DatasetSpec(
        name="lung",
        access=Access.OPEN,
        url="https://vincentarelbundock.github.io/Rdatasets/csv/survival/cancer.csv",
        sha256="4045e3fee76936bb8bd9312243d7b81b36ed224a12800fcef12e8a361883cfb6",
        license="LGPL-2.1-or-later (R survival via Rdatasets)",
        citation=(
            "Loprinzi, C. L. et al. (1994). Prospective evaluation of "
            "prognostic variables from patient-completed questionnaires. "
            "North Central Cancer Treatment Group. JCO 12(3), 601-607."
        ),
        description=(
            "NCCTG advanced lung-cancer cohort, 228 patients, 165 deaths. "
            "Endpoint is death in days. Covariates include institution, "
            "ECOG/Karnofsky performance scores (physician and patient "
            "ratings), calorie intake, weight loss. Missingness in several "
            "covariates is real and preserved."
        ),
        parser=parse_lung,
        tags=("clinical",),
        time_unit="days",
    ),
    "veteran": DatasetSpec(
        name="veteran",
        access=Access.OPEN,
        url="https://vincentarelbundock.github.io/Rdatasets/csv/survival/veteran.csv",
        sha256="3fba5cb9b15a10ab94d54e28b54d2c39d95c27b0fcff1c57aee0eb7edfd93950",
        license="LGPL-2.1-or-later (R survival via Rdatasets)",
        citation=(
            "Kalbfleisch, J. D. & Prentice, R. L. (2002). The Statistical "
            "Analysis of Failure Time Data, 2nd ed. Appendix A, Veterans' "
            "Administration lung cancer trial."
        ),
        description=(
            "Veterans Administration lung-cancer trial, 137 patients with "
            "128 deaths. Endpoint is death in days. Covariates: treatment "
            "arm (standard vs test), cell type (squamous/smallcell/adeno/"
            "large), Karnofsky score, months from diagnosis, age, prior "
            "therapy. Classic worked example for AFT and the "
            "proportional-hazards diagnostic in Kalbfleisch & Prentice."
        ),
        parser=parse_veteran,
        tags=("clinical",),
        time_unit="days",
    ),
    "melanoma": DatasetSpec(
        name="melanoma",
        access=Access.OPEN,
        url="https://vincentarelbundock.github.io/Rdatasets/csv/boot/melanoma.csv",
        sha256="a88058334680d06bb9bac458ca806f1eb67c57109a2107f4d11a4aef7476baa6",
        license="GPL-3 (R boot package via Rdatasets)",
        citation=(
            "Drzewiecki, K. T. & Andersen, P. K. (1982). Survival with "
            "malignant melanoma: a regression analysis of prognostic "
            "factors. Cancer 49(11), 2414-2419. Also Andersen, Borgan, "
            "Gill & Keiding (1993), Statistical Models Based on Counting "
            "Processes, Springer."
        ),
        description=(
            "University Hospital of Odense melanoma cohort, 205 patients "
            "with radical surgery between 1962 and 1977. Two competing "
            "endpoints: death from melanoma (cause 1, 57 events) and "
            "death from other causes (cause 2, 14 events); 134 patients "
            "censored alive. Covariates: sex, age at operation, year of "
            "operation, tumour thickness (mm), ulceration indicator. "
            "Source ``status`` is recoded from {1=melanoma death, "
            "2=alive, 3=other death} to the standard {0=censored, "
            "1..K=cause} form."
        ),
        parser=parse_melanoma,
        tags=("clinical", "competing-risks"),
        time_unit="days",
    ),
    "mgus2": DatasetSpec(
        name="mgus2",
        access=Access.OPEN,
        url="https://vincentarelbundock.github.io/Rdatasets/csv/survival/mgus2.csv",
        sha256="5138ec8d8477f031f60e475c11b4fa6b048b1544682afb4fb675af74a5551567",
        license="LGPL-2.1-or-later (R survival via Rdatasets)",
        citation=(
            "Kyle, R. A. et al. (2002). A long-term study of prognosis in "
            "monoclonal gammopathy of undetermined significance. NEJM "
            "346(8), 564-569."
        ),
        description=(
            "Mayo Clinic cohort of 1384 patients with monoclonal "
            "gammopathy of undetermined significance, followed for two "
            "competing endpoints: progression to a plasma-cell malignancy "
            "(cause 1, 115 events) and death from other causes (cause 2, "
            "860 events); 409 patients were censored. The R survival "
            "source stores progression and death as parallel "
            "(time, indicator) pairs; the loader recodes to first-event-"
            "wins competing-risks form."
        ),
        parser=parse_mgus2,
        tags=("clinical", "competing-risks"),
        time_unit="months",
    ),
    "colon": DatasetSpec(
        name="colon",
        access=Access.OPEN,
        url="https://vincentarelbundock.github.io/Rdatasets/csv/survival/colon.csv",
        sha256="6f3472a64f696e3195daa198f054180c3e4c66408f7fb8c548c6f4c7b8f898ee",
        license="LGPL-2.1-or-later (R survival via Rdatasets)",
        citation=(
            "Moertel, C. G. et al. (1990). Levamisole and fluorouracil for "
            "adjuvant therapy of resected colon carcinoma. NEJM 322(6), "
            "352-358."
        ),
        description=(
            "Colon-cancer adjuvant-therapy trial, 929 patients across "
            "three treatment arms (observation, levamisole, "
            "levamisole+5FU). The R survival source carries two rows per "
            "subject (recurrence and death endpoints); the loader pivots "
            "to wide form and recodes to first-event-wins competing risks: "
            "cause 1 = recurrence (468 events), cause 2 = death without "
            "prior recurrence (38 events), cause 0 = censored disease-"
            "free (423 patients). Time is in days from randomisation."
        ),
        parser=parse_colon,
        tags=("clinical", "competing-risks"),
        time_unit="days",
    ),
    "kidney_transplant": DatasetSpec(
        name="kidney_transplant",
        access=Access.OPEN,
        url=(
            "https://raw.githubusercontent.com/CamDavidsonPilon/lifelines/"
            "master/lifelines/datasets/kidney_transplant.csv"
        ),
        sha256="5e276e73eb144cba9725b4353a09a81f88b80db6fdbfe4ac841ca4a2f54eb91c",
        license="MIT (via lifelines)",
        citation=(
            "United Network for Organ Sharing (UNOS) registry, summarised "
            "in Klein, J. P. & Moeschberger, M. L. (2003). Survival "
            "Analysis: Techniques for Censored and Truncated Data, "
            "2nd ed., Springer."
        ),
        description=(
            "863 kidney-transplant recipients followed for death. 140 "
            "deaths observed. Race-by-sex strata appear pre-one-hot-"
            "encoded as ``black_male``, ``white_male``, ``black_female`` "
            "with white_female as the dropped reference; ``age`` is the "
            "continuous covariate."
        ),
        parser=parse_kidney_transplant,
        tags=("clinical",),
        time_unit="days",
    ),
    "larynx": DatasetSpec(
        name="larynx",
        access=Access.OPEN,
        url="https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/larynx.csv",
        sha256="90113f673525f034ecd5a7b5c4e86fe736a530b4d2f0c82495690c9dd10fb402",
        license="GPL-3 (R KMsurv via Rdatasets)",
        citation=(
            "Kardaun, O. (1983). Statistical analysis of male larynx-"
            "cancer patients - a case study. Statistical Nederlandica "
            "37(3), 103-126. Compiled by Klein & Moeschberger (2003)."
        ),
        description=(
            "90 male patients diagnosed with cancer of the larynx, "
            "stratified by stage (I-IV). 50 deaths observed. Covariates: "
            "stage (1-4), age at diagnosis, year of diagnosis. Klein-"
            "Moeschberger Chapter 1 worked example for stage-stratified "
            "survival."
        ),
        parser=parse_larynx,
        tags=("clinical",),
        time_unit="months",
    ),
    "nwtco": DatasetSpec(
        name="nwtco",
        access=Access.OPEN,
        url="https://vincentarelbundock.github.io/Rdatasets/csv/survival/nwtco.csv",
        sha256="2484933ed730f05f73fbce6af476a205fb29fef1b7ad75cd966a1a835161fa7a",
        license="LGPL-2.1-or-later (R survival via Rdatasets)",
        citation=(
            "Breslow, N. E. & Chatterjee, N. (1999). Design and analysis "
            "of two-phase studies with binary outcome applied to Wilms "
            "tumour prognosis. Applied Statistics 48(4), 457-468."
        ),
        description=(
            "National Wilms Tumor Study cohort, 4028 children with 571 "
            "relapses. Endpoint is days to relapse. The ``in.subcohort`` "
            "flag marks the 668-patient case-cohort subsample used in "
            "two-phase / IPW analyses (kept as a covariate so users can "
            "subset). Covariates: histology, stage, study, age, "
            "institution-type indicator."
        ),
        parser=parse_nwtco,
        tags=("clinical",),
        time_unit="days",
    ),
    "support": DatasetSpec(
        name="support",
        access=Access.OPEN,
        url="https://hbiostat.org/data/repo/support2csv.zip",
        sha256="8ed43980742a18e1847a8dfc5530bc4b30564ad9e4ad1b1b50bbc5d29d8c86fe",
        license="hbiostat.org Vanderbilt (freely available for research and education)",
        citation=(
            "Knaus, W. A. et al. (1995). The SUPPORT prognostic model: "
            "objective estimates of survival for seriously ill hospitalized "
            "adults. Annals of Internal Medicine 122(3), 191-203."
        ),
        description=(
            "SUPPORT study cohort, 9105 seriously ill hospitalised adults "
            "across five US medical centres (1989-1994), 6201 deaths. "
            "Endpoint: ``d.time`` (days from study entry to death). The "
            "loader keeps 45 baseline and during-stay features; several of "
            "them leak the outcome under a baseline-prediction setup and "
            "users should drop them as needed: ``hospdead``, ``slos``, "
            "``charges``, ``totcst``, ``totmcst``, ``surv2m``, ``surv6m``, "
            "``prg2m``, ``prg6m``, ``sfdm2``, ``dnr``, ``dnrday``, "
            "``avtisst``. Standard ML survival benchmarks (DeepSurv, "
            "pycox) use a 14-column subset of the remainder."
        ),
        parser=parse_support,
        tags=("clinical",),
        time_unit="days",
    ),
    "telco_churn": DatasetSpec(
        name="telco_churn",
        access=Access.OPEN,
        url=(
            "https://raw.githubusercontent.com/IBM/telco-customer-churn-"
            "on-icp4d/master/data/Telco-Customer-Churn.csv"
        ),
        sha256="16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91",
        license="Apache-2.0 (IBM Watson Analytics sample)",
        citation=(
            "IBM Watson Analytics (2015). Telco Customer Churn sample "
            "dataset. The de-facto churn-survival benchmark in industry "
            "tutorials."
        ),
        description=(
            "7043 telecom subscribers from a public IBM Watson sample. "
            "Endpoint is months of service (``tenure``); ``Churn`` is the "
            "Yes/No event indicator (1869 churns observed). Covariates "
            "cover demographics, service plans, and billing. Many "
            "string-typed categorical covariates are preserved as-is; "
            "``TotalCharges`` carries blanks for zero-tenure customers, "
            "which the loader maps to nulls."
        ),
        parser=parse_telco_churn,
        tags=("churn",),
        time_unit="months",
    ),
    "tongue": DatasetSpec(
        name="tongue",
        access=Access.OPEN,
        url="https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/tongue.csv",
        sha256="38950e1cd853e1fd390555620657edfc2b8ef87717e9feb12e5066f645563b85",
        license="GPL-3 (R KMsurv via Rdatasets)",
        citation=(
            "Sickle-Santanello, B. J. et al. (1988). A reproducible "
            "system of flow-cytometric DNA analysis of paraffin-embedded "
            "solid tumours. Cytometry 9(6), 594-599. Compiled by Klein "
            "& Moeschberger (2003)."
        ),
        description=(
            "80 patients with squamous-cell carcinoma of the tongue, "
            "stratified by tumour DNA profile: aneuploid (``type=1``, "
            "n=52, 31 deaths) vs diploid (``type=2``, n=28, 22 deaths). "
            "Time in weeks. Klein-Moeschberger Chapter 1 two-sample KM "
            "example."
        ),
        parser=parse_tongue,
        tags=("clinical",),
        time_unit="weeks",
    ),
    "waltons": DatasetSpec(
        name="waltons",
        access=Access.OPEN,
        url=(
            "https://raw.githubusercontent.com/CamDavidsonPilon/lifelines/"
            "master/lifelines/datasets/waltons_dataset.csv"
        ),
        sha256="15de145fab46631806857b47dd3f92f35c081886d5a21bf9ca89b86b0d245cbf",
        license="MIT (via lifelines)",
        citation=(
            "Bundled with lifelines (Davidson-Pilon). Documented as a "
            "two-sample Kaplan-Meier teaching example."
        ),
        description=(
            "163 subjects assigned to two strata (``miR-137`` and "
            "``control``). 156 events; very few right-censored. The "
            "canonical two-sample log-rank / KM demonstration in "
            "lifelines tutorials. Heavily used in survival-tutorial "
            "code, including the lifelines documentation itself."
        ),
        parser=parse_waltons,
        tags=("clinical",),
        time_unit="days",
    ),
    "flchain": DatasetSpec(
        name="flchain",
        access=Access.OPEN,
        url="https://vincentarelbundock.github.io/Rdatasets/csv/survival/flchain.csv",
        sha256="a96bcc58addb4c127e5012c5974aec8c7daf66123ddb104f309a78f158daa563",
        license="LGPL-2.1-or-later (R survival via Rdatasets)",
        citation=(
            "Dispenzieri, A. et al. (2012). Use of nonclonal serum "
            "immunoglobulin free light chains to predict overall survival "
            "in the general population. Mayo Clinic Proceedings 87(6)."
        ),
        description=(
            "Mayo Clinic population cohort of free-light-chain assays, "
            "7874 subjects 50+ with 2169 deaths. Endpoint is death in "
            "days. Covariates include age, sex, kappa/lambda free light "
            "chains, FLC group, creatinine, MGUS indicator. Cause-of-"
            "death codes (`chapter`) are dropped here because they leak "
            "the outcome under single-event analysis; a future competing-"
            "risks loader will expose them."
        ),
        parser=parse_flchain,
        tags=("clinical",),
        time_unit="days",
    ),
    "gbsg": DatasetSpec(
        name="gbsg",
        access=Access.OPEN,
        url="https://vincentarelbundock.github.io/Rdatasets/csv/survival/gbsg.csv",
        sha256="9fa0fef0575d04d3273869be28405307f1b02993751498e8a0c844541cccd5d9",
        license="LGPL-2.1-or-later (R survival via Rdatasets)",
        citation=(
            "Schumacher, M. et al. (1994). Randomized 2x2 trial evaluating "
            "hormonal treatment and the duration of chemotherapy in node-"
            "positive breast cancer patients. JCO 12(10), 2086-2093."
        ),
        description=(
            "German Breast Cancer Study Group cohort, 686 node-positive "
            "breast cancer patients with 299 recurrence-or-death events. "
            "Endpoint: recurrence-free survival in days. Covariates: age, "
            "menopausal status, tumour size and grade, positive lymph "
            "nodes, progesterone and oestrogen receptors, hormonal "
            "treatment indicator."
        ),
        parser=parse_gbsg,
        tags=("clinical",),
        time_unit="days",
    ),
}



def _merge(target: dict[str, DatasetSpec], extra: dict[str, DatasetSpec]) -> None:
    """Add ``extra`` to ``target``, refusing to silently shadow a name."""
    clash = sorted(set(target) & set(extra))
    if clash:
        raise RuntimeError(f"duplicate dataset names across spec modules: {clash}")
    target.update(extra)


def _install_extra_specs() -> None:
    """Merge the out-of-module spec tables into the registry.

    Imported here rather than at module scope: the spec modules import
    parsers, which import nothing from the registry, but variants are built
    *from* the core entries above and so must run after they exist.
    """
    from tausurv.datasets import _specs_bench, _specs_r, _specs_variants  # noqa: PLC0415

    _merge(_REGISTRY, _specs_r.SPECS)
    _merge(_REGISTRY, _specs_bench.SPECS)
    _merge(_REGISTRY, _specs_variants.build(_REGISTRY))


_install_extra_specs()


#: Names the same table travels under elsewhere in the literature, mapped to
#: the canonical registry name. Aliases resolve in :func:`load_dataset` and
#: :func:`dataset_info` but are deliberately absent from
#: :func:`list_datasets`, so that every table has exactly one name here and
#: results reported against it are unambiguous.
_ALIASES: dict[str, str] = {
    # GBSG2 in Rdatasets / TH.data is the 686-patient German trial, which is
    # what `gbsg` already is. The 2232-row Rotterdam combination that
    # deep-survival papers *also* call GBSG is `gbsg:deepsurv`.
    "gbsg2": "gbsg",
    # The DeepSurv GBSG table is a Rotterdam + German-trial combination, so
    # it is its own study rather than a `gbsg:` variant. Both the name the
    # deep-survival literature uses and the variant spelling resolve to it.
    "gbsg_deepsurv": "gbsg_rotterdam",
    "gbsg:deepsurv": "gbsg_rotterdam",
    "whas:deepsurv": "whas_deepsurv",
    "actg320": "aids",
    "whas": "whas500",
    # R survival calls the retinopathy trial `diabetic`; KMsurv and asaur
    # call the Channing House cohort `ChanningHouse`.
    "diabetic": "retinopathy",
    "channinghouse": "channing",
    "kidtran": "kidney_transplant",
    # R survival's `cancer` and `lung` are the same NCCTG cohort.
    "cancer": "lung",
    "nafld1": "nafld",
    "udca1": "udca",
    "leukemia": "aml",
    "hodg": "hodgkins",
    "prostatesurvival": "prostate",
    "hepatocellularcarcinoma": "hepatocellular",
    "pharmacosmoking": "pharmaco_smoking",
    "valveseat": "valve_seat",
    "support2": "support",
}


def resolve_name(name: str) -> str:
    """Canonical registry name for ``name``, following aliases.

    Matching is case-insensitive and tolerant of ``-`` for ``_``, because
    dataset names reach this function from command lines as often as from
    code. An alias on the base part carries its variant through, so
    ``"gbsg2"`` and ``"actg320:death"`` both resolve.
    """
    if name in _REGISTRY:
        return name
    key = name.strip().lower().replace("-", "_")
    if key in _REGISTRY:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    base, variant = split_name(key)
    if variant is not None and base in _ALIASES:
        candidate = f"{_ALIASES[base]}:{variant}"
        if candidate in _REGISTRY:
            return candidate
    return name


def list_datasets(
    tag: str | None = None,
    *,
    base_only: bool = False,
    access: Access | str | None = None,
) -> list[str]:
    """Sorted registry names.

    Args:
        tag: keep only datasets carrying this tag. Common tags:
            ``"clinical"``, ``"reliability"``, ``"competing-risks"``,
            ``"benchmark"``, ``"left-truncated"``, ``"rare-events"``.
        base_only: drop ``base:variant`` entries, leaving one name per
            study. Useful for "show me what cohorts exist" as opposed to
            "show me every table I could load".
        access: keep only datasets at this access level. Pass
            ``Access.OPEN`` for the set that needs no manual setup, which
            is the set a test suite can cover.

    Aliases are never returned; :func:`resolve_name` accepts them.
    """
    if access is not None:
        access = Access(access)
    names = []
    for name, spec in _REGISTRY.items():
        if tag is not None and tag not in spec.tags:
            continue
        if base_only and spec.is_variant:
            continue
        if access is not None and spec.access is not access:
            continue
        names.append(name)
    return sorted(names)


def list_variants(name: str) -> list[str]:
    """Every registered reading of the study ``name`` belongs to, including it."""
    base = split_name(resolve_name(name))[0]
    return sorted(
        n for n in _REGISTRY if n == base or n.startswith(f"{base}:")
    )


def dataset_info(name: str) -> DatasetInfo:
    """Metadata for a dataset without triggering a download."""
    spec = _get_spec(name)
    return DatasetInfo(
        name=spec.name,
        access=spec.access,
        license=spec.license,
        citation=spec.citation,
        description=spec.description,
        url=spec.url,
        sha256=spec.sha256,
        tags=spec.tags,
        time_unit=spec.time_unit,
        requires=spec.requires,
    )


def load_dataset(
    name: str,
    *,
    path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> SurvivalBunch:
    """Load a dataset, downloading and caching it on first use.

    Args:
        name: registry name or alias; see :func:`list_datasets`. Accepts
            ``base:variant`` -- :func:`list_variants` shows what a study
            offers.
        path: local directory (or file) holding the data. Required for
            ``USER_PROVIDED`` and ``CREDENTIALED`` datasets, which are
            never fetched; ignored for open ones.
        cache_dir: override the cache root. Defaults to the ``TAUSURV_DATA``
            env var, then ``platformdirs.user_cache_dir("tausurv")/datasets``.
        force_download: re-download even if the digest is already cached.
            Useful after a registry hash bump.

    Raises:
        UnknownDatasetError: no such name, with near-miss suggestions.
        MissingDependencyError: the parser needs an optional dependency.
        UserProvidedDatasetError: redistribution-restricted and no ``path``.
        CredentialedDatasetError: needs credentials and no ``path``.
    """
    spec = _get_spec(name)
    if spec.requires is not None:
        _require(spec)
    if path is not None:
        return spec.parser(Path(path).expanduser(), spec)
    if spec.access is Access.CREDENTIALED:
        raise CredentialedDatasetError(spec.name, spec.access_help or "")
    if spec.access is Access.USER_PROVIDED:
        raise UserProvidedDatasetError(spec.name, spec.access_help or "")
    assert spec.url is not None and spec.sha256 is not None
    cached = fetch_to_cache(
        spec.name,
        spec.url,
        spec.sha256,
        cache_dir=cache_dir,
        force_download=force_download,
    )
    return spec.parser(cached, spec)


def _require(spec: DatasetSpec) -> None:
    """Fail early, with install instructions, if an optional dependency is absent."""
    import importlib.util  # noqa: PLC0415

    assert spec.requires is not None
    if importlib.util.find_spec(spec.requires) is None:
        raise MissingDependencyError(spec.name, spec.requires, "deepsurv")


def _get_spec(name: str) -> DatasetSpec:
    resolved = resolve_name(name)
    spec = _REGISTRY.get(resolved)
    if spec is None:
        raise UnknownDatasetError(name, list(_REGISTRY))
    return spec
