"""Registry entries for the R-ecosystem cohorts (Rdatasets mirrors).

Split out from :mod:`tausurv.datasets._registry` purely for length. The
registry merges these in; nothing here is resolved directly.
"""

from __future__ import annotations

from tausurv.datasets import _parsers_r as p
from tausurv.datasets._spec import Access, DatasetSpec

_R = "https://vincentarelbundock.github.io/Rdatasets/csv"
_SURVIVAL_LICENSE = "LGPL-2.1-or-later (R survival via Rdatasets)"
_KMSURV_LICENSE = "GPL-3 (R KMsurv via Rdatasets)"
_ASAUR_LICENSE = "GPL-2 (R asaur via Rdatasets)"

_KM_BOOK = (
    "Klein, J. P. & Moeschberger, M. L. (2003). Survival Analysis: "
    "Techniques for Censored and Truncated Data, 2nd ed. Springer."
)
_MOORE_BOOK = (
    "Moore, D. F. (2016). Applied Survival Analysis Using R. Springer."
)

SPECS: dict[str, DatasetSpec] = {
    # ---- breast cancer: the Rotterdam cohort and its two endpoints ----
    "rotterdam": DatasetSpec(
        name="rotterdam",
        access=Access.OPEN,
        url=f"{_R}/survival/rotterdam.csv",
        sha256="62703670d3be5d49c5476fb7131b020bdca9f1c25bb4522fd6abe8f9dab61f1b",
        license=_SURVIVAL_LICENSE,
        citation=(
            "Royston, P. & Altman, D. G. (2013). External validation of a "
            "Cox prognostic model: principles and methods. BMC Medical "
            "Research Methodology 13, 33."
        ),
        description=(
            "Rotterdam tumour bank, 2982 primary breast cancer patients "
            "treated 1978-1993, with overall survival as the endpoint "
            "(1272 deaths). Covariates: year of surgery, age, menopausal "
            "status, tumour size and grade, positive nodes, progesterone "
            "and oestrogen receptors, hormonal and chemo treatment. The "
            "cohort records recurrence as well as death; this entry uses "
            "death, `rotterdam:recurrence` uses recurrence, and "
            "`rotterdam:rfs` uses the earlier of the two. Whichever "
            "endpoint is unused is dropped from the covariates."
        ),
        parser=p.parse_rotterdam,
        tags=("clinical", "breast-cancer"),
        time_unit="days",
    ),
    "rotterdam:recurrence": DatasetSpec(
        name="rotterdam:recurrence",
        access=Access.OPEN,
        url=f"{_R}/survival/rotterdam.csv",
        sha256="62703670d3be5d49c5476fb7131b020bdca9f1c25bb4522fd6abe8f9dab61f1b",
        license=_SURVIVAL_LICENSE,
        citation=(
            "Royston, P. & Altman, D. G. (2013). External validation of a "
            "Cox prognostic model: principles and methods. BMC Medical "
            "Research Methodology 13, 33."
        ),
        description=(
            "Rotterdam tumour bank with time to disease recurrence as the "
            "endpoint (1518 recurrences in 2982 patients). Death without "
            "recurrence is treated as censoring here; for the competing "
            "reading use `rotterdam:rfs`, which counts either event."
        ),
        parser=p.parse_rotterdam_recurrence,
        tags=("clinical", "breast-cancer"),
        time_unit="days",
    ),
    "rotterdam:rfs": DatasetSpec(
        name="rotterdam:rfs",
        access=Access.OPEN,
        url=f"{_R}/survival/rotterdam.csv",
        sha256="62703670d3be5d49c5476fb7131b020bdca9f1c25bb4522fd6abe8f9dab61f1b",
        license=_SURVIVAL_LICENSE,
        citation=(
            "Royston, P. & Altman, D. G. (2013). External validation of a "
            "Cox prognostic model: principles and methods. BMC Medical "
            "Research Methodology 13, 33."
        ),
        description=(
            "Rotterdam tumour bank with recurrence-free survival: the event "
            "is whichever of recurrence or death comes first, and the time "
            "is that first event. This is the endpoint Royston and Altman "
            "model, and the one the DeepSurv `gbsg:deepsurv` benchmark "
            "inherits, so it is the right table to compare deep-survival "
            "results against."
        ),
        parser=p.parse_rotterdam_rfs,
        tags=("clinical", "breast-cancer", "benchmark"),
        time_unit="days",
    ),
    # ---- other single-event clinical cohorts ----
    "retinopathy": DatasetSpec(
        name="retinopathy",
        access=Access.OPEN,
        url=f"{_R}/survival/retinopathy.csv",
        sha256="b3152f26c7b0c52a990c0e97aa5d47b2a288c7ad6d7d06a8126567d3055ec656",
        license=_SURVIVAL_LICENSE,
        citation=(
            "Huster, W. J., Brookmeyer, R. & Self, S. G. (1989). Modelling "
            "paired survival data with covariates. Biometrics 45(1), "
            "145-156."
        ),
        description=(
            "Diabetic Retinopathy Study: 197 patients with one eye "
            "randomised to laser photocoagulation and the other left "
            "untreated, giving 394 paired observations of time to severe "
            "vision loss (155 events). The pairing is the point of the "
            "dataset -- rows are not independent, and it is the standard "
            "test case for clustered and frailty models."
        ),
        parser=p.parse_retinopathy,
        tags=("clinical", "clustered"),
        time_unit="months",
    ),
    "ovarian": DatasetSpec(
        name="ovarian",
        access=Access.OPEN,
        url=f"{_R}/survival/ovarian.csv",
        sha256="4fe12ec9566848fe9760fb0d1d2ed3ded1878ee1f3250ca90c0d0e1e4dd60d63",
        license=_SURVIVAL_LICENSE,
        citation=(
            "Edmunson, J. H. et al. (1979). Different chemotherapeutic "
            "sensitivities and host factors affecting prognosis in advanced "
            "ovarian carcinoma versus minimal residual disease. Cancer "
            "Treatment Reports 63, 241-247."
        ),
        description=(
            "26 patients with advanced ovarian carcinoma randomised between "
            "two chemotherapy regimens, 12 deaths. Four covariates: age, "
            "residual disease, treatment, ECOG performance status. Small "
            "enough to check an implementation by hand, which is what it is "
            "usually used for."
        ),
        parser=p.parse_ovarian,
        tags=("clinical",),
        time_unit="days",
    ),
    "stanford2": DatasetSpec(
        name="stanford2",
        access=Access.OPEN,
        url=f"{_R}/survival/stanford2.csv",
        sha256="c76128bc01eba8f9794d154231d79e789fdebe05b29024cf8461fbed3e502d7b",
        license=_SURVIVAL_LICENSE,
        citation=(
            "Escobar, L. A. & Meeker, W. Q. (1992). Assessing influence in "
            "regression analysis with censored data. Biometrics 48, "
            "507-528."
        ),
        description=(
            "184 Stanford heart transplant recipients followed for survival "
            "after transplant (113 deaths). Covariates are age and T5, a "
            "tissue mismatch score. The canonical example for influence "
            "diagnostics and for non-monotone covariate effects: survival "
            "is worst at both extremes of age."
        ),
        parser=p.parse_stanford2,
        tags=("clinical",),
        time_unit="days",
    ),
    "myeloma": DatasetSpec(
        name="myeloma",
        access=Access.OPEN,
        url=f"{_R}/survival/myeloma.csv",
        sha256="670f441970109e736142dd512bc448cdd339e6f9d5e098b6aec18c53c1b5a27d",
        license=_SURVIVAL_LICENSE,
        citation=(
            "Krall, J. M., Uthoff, V. A. & Harley, J. B. (1975). A step-up "
            "procedure for selecting variables associated with survival. "
            "Biometrics 31, 49-57."
        ),
        description=(
            "3882 patients with multiple myeloma seen at the Mayo Clinic, "
            "2769 deaths -- a 71% event rate over a large cohort, which is "
            "unusual and makes it useful for methods that struggle when "
            "censoring is light. `entry` is the delay from diagnosis to "
            "registration and is a left-truncation time, not a covariate."
        ),
        parser=p.parse_myeloma,
        tags=("clinical", "left-truncated"),
        time_unit="days",
    ),
    "nafld": DatasetSpec(
        name="nafld",
        access=Access.OPEN,
        url=f"{_R}/survival/nafld1.csv",
        sha256="5b59129b36f6c600d065a042e1f93fefa0940685e5e149a3ef92f731b0572444",
        license=_SURVIVAL_LICENSE,
        citation=(
            "Allen, A. M. et al. (2018). Nonalcoholic fatty liver disease "
            "incidence and impact on metabolic burden and death: a 20-year "
            "community study. Hepatology 67(5), 1726-1736."
        ),
        description=(
            "17549 residents of Olmsted County, Minnesota: every adult "
            "diagnosed with non-alcoholic fatty liver disease 1997-2014 "
            "plus four matched controls each, followed for death (1364 "
            "events, 7.8%). The largest open cohort in the registry and the "
            "one closest to a real epidemiological sample -- large, sparse "
            "in events, and matched rather than randomised."
        ),
        parser=p.parse_nafld,
        tags=("clinical", "epidemiology"),
        time_unit="days",
    ),
    "udca": DatasetSpec(
        name="udca",
        access=Access.OPEN,
        url=f"{_R}/survival/udca1.csv",
        sha256="d67d72eaefe15a05038e0550b91ff361dc32fb3eab0ba17738517a95f6fec686",
        license=_SURVIVAL_LICENSE,
        citation=(
            "Lindor, K. D. et al. (1994). Ursodeoxycholic acid in the "
            "treatment of primary biliary cirrhosis. Gastroenterology "
            "106(5), 1284-1290."
        ),
        description=(
            "170 patients in a Mayo Clinic trial of ursodeoxycholic acid "
            "for primary biliary cirrhosis, 72 events. Covariates: "
            "treatment arm, histologic stage, bilirubin, and the Mayo risk "
            "score. A small randomised trial with a strong treatment "
            "effect, which makes it a clean test of whether a model "
            "recovers one."
        ),
        parser=p.parse_udca,
        tags=("clinical", "trial"),
        time_unit="days",
    ),
    "aml": DatasetSpec(
        name="aml",
        access=Access.OPEN,
        url=f"{_R}/survival/aml.csv",
        sha256="f0423f543f01227152eecd89911e45c09bf328d78dc63dfc060f6bf412cf7985",
        license=_SURVIVAL_LICENSE,
        citation="Miller, R. G. (1981). Survival Analysis. John Wiley & Sons.",
        description=(
            "23 patients with acute myelogenous leukemia, randomised to "
            "continue or discontinue maintenance chemotherapy after "
            "remission (18 relapses). The smallest dataset in the registry "
            "and the two-sample log-rank example that opens most survival "
            "courses."
        ),
        parser=p.parse_aml,
        tags=("clinical", "trial"),
        time_unit="weeks",
    ),
    "channing": DatasetSpec(
        name="channing",
        access=Access.OPEN,
        url=f"{_R}/KMsurv/channing.csv",
        sha256="bef06d7c7031c888765c94e9e34d3e9ef1226a1ae51486a12effb2fd775269b1",
        license=_KMSURV_LICENSE,
        citation=(
            "Hyde, J. (1980). Testing survival with incomplete "
            "observations. In: Biostatistics Casebook. John Wiley & Sons."
        ),
        description=(
            "462 residents of the Channing House retirement community, 176 "
            "deaths. The standard left-truncation example: residents enter "
            "the cohort at the age they move in, so anyone who died younger "
            "than that age was never eligible to be observed. Ignoring the "
            "truncation inflates estimated survival badly. `ageentry` is "
            "kept as a covariate so both analyses are possible."
        ),
        parser=p.parse_channing,
        tags=("clinical", "left-truncated"),
        time_unit="months",
    ),
    "pharmaco_smoking": DatasetSpec(
        name="pharmaco_smoking",
        access=Access.OPEN,
        url=f"{_R}/asaur/pharmacoSmoking.csv",
        sha256="4b60b05350215f0801d0de8f8c69d83808295e1f8964e5bd5901b133cd756744",
        license=_ASAUR_LICENSE,
        citation=(
            "Steinberg, M. B. et al. (2009). Triple-combination "
            "pharmacotherapy for medically ill smokers: a randomized "
            "trial. Annals of Internal Medicine 150(7), 447-454."
        ),
        description=(
            "125 smokers randomised between triple-combination "
            "pharmacotherapy and patch alone, followed for time to relapse "
            "(89 relapses). Covariates include age, sex, race, employment, "
            "years smoking, and prior quit attempts. Moore's running "
            "example for model building and for coding categorical "
            "predictors."
        ),
        parser=p.parse_pharmaco_smoking,
        tags=("clinical", "trial", "behavioural"),
        time_unit="days",
    ),
    "gastric_xelox": DatasetSpec(
        name="gastric_xelox",
        access=Access.OPEN,
        url=f"{_R}/asaur/gastricXelox.csv",
        sha256="2653504109df55c3b9259bd2ec6088c2269e9e27a665620c35cdd8e6f0a0e438",
        license=_ASAUR_LICENSE,
        citation=_MOORE_BOOK,
        description=(
            "48 patients with advanced gastric cancer treated with the "
            "Xelox regimen, followed for progression-free survival in weeks "
            "(32 events). No covariates beyond the endpoint -- it is a "
            "one-sample dataset, used for fitting and comparing parametric "
            "survival distributions."
        ),
        parser=p.parse_gastric_xelox,
        tags=("clinical", "single-arm"),
        time_unit="weeks",
    ),
    "ashkenazi": DatasetSpec(
        name="ashkenazi",
        access=Access.OPEN,
        url=f"{_R}/asaur/ashkenazi.csv",
        sha256="7e602ce572e24861cc2bda325df7a031f59a742ad3c067262a5f86c43e41d615",
        license=_ASAUR_LICENSE,
        citation=(
            "Struewing, J. P. et al. (1997). The risk of cancer associated "
            "with specific mutations of BRCA1 and BRCA2 among Ashkenazi "
            "Jews. New England Journal of Medicine 336(20), 1401-1408."
        ),
        description=(
            "3920 Ashkenazi Jewish women, 473 with breast cancer, genotyped "
            "for BRCA1/BRCA2 founder mutations. The time scale is age, not "
            "follow-up duration, so the endpoint is age at diagnosis and "
            "everyone is at risk from birth. Relatives are grouped by "
            "`famID`, so rows cluster within families."
        ),
        parser=p.parse_ashkenazi,
        tags=("clinical", "genetics", "clustered"),
        time_unit="years",
    ),
    # ---- competing risks ----
    "transplant": DatasetSpec(
        name="transplant",
        access=Access.OPEN,
        url=f"{_R}/survival/transplant.csv",
        sha256="5f8d031eecce3d9f8fc5e67c2697d5d965c3009ecf7f66c03d9581b2098071c5",
        license=_SURVIVAL_LICENSE,
        citation=(
            "Therneau, T. M. (2024). A Package for Survival Analysis in R. "
            "R package version 3.7."
        ),
        description=(
            "815 patients on a liver transplant waiting list, with three "
            "ways to leave it: transplant (636), death on the list (66), "
            "and withdrawal (37). Only 76 remain censored. The clearest "
            "argument in the registry against treating competing events as "
            "censoring -- do that here and the estimated transplant rate "
            "exceeds one."
        ),
        parser=p.parse_transplant,
        tags=("clinical", "competing-risks"),
        time_unit="days",
    ),
    "hoel": DatasetSpec(
        name="hoel",
        access=Access.OPEN,
        url=f"{_R}/survival/hoel.csv",
        sha256="43af66e7c3b281db09d4e08c58ce222a1bab8decb1c26948196a6efbaabc2976",
        license=_SURVIVAL_LICENSE,
        citation=(
            "Hoel, D. G. (1972). A representation of mortality data by "
            "competing risks. Biometrics 28(2), 475-488."
        ),
        description=(
            "181 irradiated mice, each of which died of exactly one of "
            "thymic lymphoma, reticulum cell sarcoma, or another cause, "
            "under two germ-free conditions. There is no censoring at all: "
            "the event rate is 100% and the entire question is which cause "
            "won. The purest competing-risks illustration available."
        ),
        parser=p.parse_hoel,
        tags=("competing-risks", "preclinical"),
        time_unit="days",
    ),
    "mgus": DatasetSpec(
        name="mgus",
        access=Access.OPEN,
        url=f"{_R}/survival/mgus.csv",
        sha256="84e8aee413989ead6015367c4a8cbe8ef3aeb93b39f774cf176f9029c5b41413",
        license=_SURVIVAL_LICENSE,
        citation=(
            "Kyle, R. A. (1993). Benign monoclonal gammopathy -- after 20 "
            "to 35 years of follow-up. Mayo Clinic Proceedings 68(1), "
            "26-36."
        ),
        description=(
            "241 patients with monoclonal gammopathy of undetermined "
            "significance followed for progression to plasma cell "
            "malignancy or death, recoded first-event-wins. Kyle's earlier "
            "and smaller series; `mgus2` is the 1384-patient successor. "
            "The malignancy type is dropped because it is recorded only "
            "for progressors and therefore leaks the cause."
        ),
        parser=p.parse_mgus_competing,
        tags=("clinical", "competing-risks"),
        time_unit="days",
    ),
    "mgus:death": DatasetSpec(
        name="mgus:death",
        access=Access.OPEN,
        url=f"{_R}/survival/mgus.csv",
        sha256="84e8aee413989ead6015367c4a8cbe8ef3aeb93b39f774cf176f9029c5b41413",
        license=_SURVIVAL_LICENSE,
        citation=(
            "Kyle, R. A. (1993). Benign monoclonal gammopathy -- after 20 "
            "to 35 years of follow-up. Mayo Clinic Proceedings 68(1), "
            "26-36."
        ),
        description=(
            "The MGUS cohort with overall survival as a single endpoint: "
            "225 of 241 patients died, leaving only 6.6% censored. "
            "Progression to plasma cell malignancy is ignored rather than "
            "competing, which is the reading benchmark tables use for this "
            "cohort; `mgus` itself is the competing-risks form."
        ),
        parser=p.parse_mgus_death,
        tags=("clinical",),
        time_unit="days",
    ),
    "leukemia": DatasetSpec(
        name="leukemia",
        access=Access.OPEN,
        url=f"{_R}/MASS/gehan.csv",
        sha256="b6dc0c39f7eeec171950104432525393af3950d3f2d802439f63195c2147d8a2",
        license="GPL-2 | GPL-3 (R MASS via Rdatasets)",
        citation=(
            "Gehan, E. A. (1965). A generalized Wilcoxon test for comparing "
            "arbitrarily singly-censored samples. Biometrika 52(1-2), "
            "203-224."
        ),
        description=(
            "Freireich's 6-mercaptopurine trial as analysed by Gehan: 21 "
            "matched pairs of leukemia patients in remission, one of each "
            "pair given 6-MP and the other placebo, followed to relapse (30 "
            "relapses in 42 patients). The dataset the generalized Wilcoxon "
            "test was introduced on, and the standard paired survival "
            "example. `pair` is kept as a covariate because the matching is "
            "the design. Distinct from `aml`, R survival's 23-patient "
            "maintenance trial, which the literature also calls leukemia."
        ),
        parser=p.parse_gehan,
        tags=("clinical", "trial", "clustered"),
        time_unit="weeks",
    ),
    "myeloid": DatasetSpec(
        name="myeloid",
        access=Access.OPEN,
        url=f"{_R}/survival/myeloid.csv",
        sha256="69a293d89a1fafa06ddb0ee4d4b12ef1c7a7db6aaa6e28997c25586989e0708c",
        license=_SURVIVAL_LICENSE,
        citation=(
            "Therneau, T. M. (2024). A Package for Survival Analysis in R. "
            "R package version 3.7."
        ),
        description=(
            "646 patients in a two-arm acute myeloid leukemia trial, 320 "
            "deaths. The source also records complete response, stem cell "
            "transplant and relapse times; those are intermediate states "
            "reached after baseline, so they are dropped here rather than "
            "offered as covariates. The multi-state analysis they support "
            "is the reason the cohort is distributed."
        ),
        parser=p.parse_myeloid,
        tags=("clinical", "trial"),
        time_unit="days",
    ),
    "prostate": DatasetSpec(
        name="prostate",
        access=Access.OPEN,
        url=f"{_R}/asaur/prostateSurvival.csv",
        sha256="a992eb03dc5642145e76cb632dde0578c523b535d27b941722739fd89449d470",
        license=_ASAUR_LICENSE,
        citation=_MOORE_BOOK,
        description=(
            "14294 men diagnosed with prostate cancer, drawn from SEER and "
            "grouped by tumour grade, stage and age. 799 died of prostate "
            "cancer and 3240 of something else -- competing mortality "
            "outnumbers the event of interest four to one, which is the "
            "standard demonstration that cause-specific and "
            "subdistribution hazards answer different questions. A "
            "redistributable stand-in for the gated `seer` entry."
        ),
        parser=p.parse_prostate_survival,
        tags=("clinical", "competing-risks", "registry"),
        time_unit="months",
    ),
    "hepatocellular": DatasetSpec(
        name="hepatocellular",
        access=Access.OPEN,
        url=f"{_R}/asaur/hepatoCellular.csv",
        sha256="4b483a3f9585e3a586a67754ac4d226a968b30b055db650098fa1cb0bccc96c2",
        license=_ASAUR_LICENSE,
        citation=(
            "Li, J. et al. (2011). CXCL17 expression predicts poor "
            "prognosis and correlates with adverse immune infiltration in "
            "hepatocellular carcinoma. PLoS ONE 9(10), e110064."
        ),
        description=(
            "227 patients with hepatocellular carcinoma and 48 columns of "
            "clinical staging plus immunohistochemical marker counts, with "
            "overall survival as the endpoint (97 deaths). Wide relative to "
            "its length and heavily correlated across the marker block, "
            "which is what makes it a variable-selection example. "
            "`hepatocellular:rfs` uses recurrence instead."
        ),
        parser=p.parse_hepato_cellular,
        tags=("clinical", "biomarkers"),
        time_unit="months",
    ),
    "hepatocellular:rfs": DatasetSpec(
        name="hepatocellular:rfs",
        access=Access.OPEN,
        url=f"{_R}/asaur/hepatoCellular.csv",
        sha256="4b483a3f9585e3a586a67754ac4d226a968b30b055db650098fa1cb0bccc96c2",
        license=_ASAUR_LICENSE,
        citation=(
            "Li, J. et al. (2011). CXCL17 expression predicts poor "
            "prognosis and correlates with adverse immune infiltration in "
            "hepatocellular carcinoma. PLoS ONE 9(10), e110064."
        ),
        description=(
            "Hepatocellular carcinoma cohort with recurrence-free survival "
            "as the endpoint (143 recurrences in 227 patients). The "
            "overall-survival columns are dropped so they cannot leak."
        ),
        parser=p.parse_hepato_cellular_rfs,
        tags=("clinical", "biomarkers"),
        time_unit="months",
    ),
    # ---- recurrent-event cohorts reduced to time-to-first-event ----
    "bladder": DatasetSpec(
        name="bladder",
        access=Access.OPEN,
        url=f"{_R}/survival/bladder.csv",
        sha256="28e0cf715c419d01b9e1030bd9ef46d320621087e1e7806a61bdb13267ace055",
        license=_SURVIVAL_LICENSE,
        citation=(
            "Wei, L. J., Lin, D. Y. & Weissfeld, L. (1989). Regression "
            "analysis of multivariate incomplete failure time data by "
            "modeling marginal distributions. JASA 84(408), 1065-1073."
        ),
        description=(
            "85 patients with superficial bladder tumours, randomised "
            "between placebo and thiotepa, followed for tumour recurrence. "
            "The source is a four-recurrences-per-patient table; this entry "
            "keeps the first interval only, so it is an ordinary "
            "right-censored cohort. The recurrent-event structure it was "
            "collected for is not preserved by that reduction."
        ),
        parser=p.parse_bladder_first,
        tags=("clinical", "trial", "first-event"),
        time_unit="months",
    ),
    "cgd": DatasetSpec(
        name="cgd",
        access=Access.OPEN,
        url=f"{_R}/survival/cgd.csv",
        sha256="923b9cc7e7a44b5eac6c2e22e74f82a1d60e8c9909905371ef54460787ad9ef0",
        license=_SURVIVAL_LICENSE,
        citation=(
            "Fleming, T. R. & Harrington, D. P. (1991). Counting Processes "
            "and Survival Analysis. John Wiley & Sons."
        ),
        description=(
            "128 patients with chronic granulomatous disease in a placebo-"
            "controlled trial of gamma interferon, reduced to time to first "
            "serious infection. Covariates include treatment, inheritance "
            "pattern, age, height, weight and steroid use. The full "
            "recurrent-event table is what Fleming and Harrington analyse."
        ),
        parser=p.parse_cgd_first,
        tags=("clinical", "trial", "first-event"),
        time_unit="days",
    ),
    "kidney_catheter": DatasetSpec(
        name="kidney_catheter",
        access=Access.OPEN,
        url=f"{_R}/survival/kidney.csv",
        sha256="b41cb1521066521a92a4b887e762350762de37d3edb6435be37073933c5a3473",
        license=_SURVIVAL_LICENSE,
        citation=(
            "McGilchrist, C. A. & Aisbett, C. W. (1991). Regression with "
            "frailty in survival analysis. Biometrics 47(2), 461-466."
        ),
        description=(
            "76 catheter insertions in 38 kidney dialysis patients, "
            "followed for infection at the insertion site. Two records per "
            "patient, which is the entire point: it is the reference "
            "dataset for shared frailty models. The source `frail` column "
            "is McGilchrist and Aisbett's fitted frailty estimate, not a "
            "baseline measurement, and is dropped."
        ),
        parser=p.parse_kidney_catheter,
        tags=("clinical", "frailty", "clustered"),
        time_unit="days",
    ),
    "valve_seat": DatasetSpec(
        name="valve_seat",
        access=Access.OPEN,
        url=f"{_R}/survival/valveSeat.csv",
        sha256="d40fa794eae55c27e20957193e8b6abc247228065de52abdf0e3bc18aa320673",
        license=_SURVIVAL_LICENSE,
        citation=(
            "Nelson, W. B. & Doganaksoy, N. (1989). A Computer Program for "
            "an Estimate and Confidence Limits for the Mean Cumulative "
            "Function for Cost or Number of Repairs of Repairable Products. "
            "GE Research & Development."
        ),
        description=(
            "41 diesel engines followed for valve-seat replacement, reduced "
            "to time to first replacement. No covariates survive the "
            "reduction, so this is a one-sample reliability dataset: the "
            "question is the shape of the failure-time distribution, not "
            "which unit fails first."
        ),
        parser=p.parse_valve_seat,
        tags=("reliability", "first-event"),
        time_unit="days",
    ),
    # ---- Klein & Moeschberger textbook cohorts ----
    "bmt": DatasetSpec(
        name="bmt",
        access=Access.OPEN,
        url=f"{_R}/KMsurv/bmt.csv",
        sha256="6643c86e0251f6d00e73ea4cabe3f4b702fb67fffbbb07d073c2f383122d25ff",
        license=_KMSURV_LICENSE,
        citation=(
            "Copelan, E. A. et al. (1991). Treatment for acute "
            "myelocytic leukemia with allogeneic bone marrow "
            "transplantation following preparation with BuCy2. Blood "
            "78(3), 838-843."
        ),
        description=(
            "137 bone marrow transplant recipients across three "
            "disease-risk groups (ALL, low-risk AML, high-risk AML), with "
            "disease-free survival as the endpoint: the earlier of relapse "
            "or death in remission, 83 events. Covariates z1-z10 cover "
            "donor and recipient age and sex, CMV status, waiting time and "
            "graft-versus-host prophylaxis. `bmt:competing` separates the "
            "two ways of failing."
        ),
        parser=p.parse_bmt,
        tags=("clinical", "transplant"),
        time_unit="days",
    ),
    "bmt:competing": DatasetSpec(
        name="bmt:competing",
        access=Access.OPEN,
        url=f"{_R}/KMsurv/bmt.csv",
        sha256="6643c86e0251f6d00e73ea4cabe3f4b702fb67fffbbb07d073c2f383122d25ff",
        license=_KMSURV_LICENSE,
        citation=(
            "Copelan, E. A. et al. (1991). Treatment for acute "
            "myelocytic leukemia with allogeneic bone marrow "
            "transplantation following preparation with BuCy2. Blood "
            "78(3), 838-843."
        ),
        description=(
            "The same 137 transplant recipients, with relapse (42) and "
            "death in remission (41) as competing causes rather than pooled "
            "into disease-free survival. They genuinely compete: a relapse "
            "cannot be observed after a treatment-related death, and the "
            "two have opposite relationships with graft-versus-host "
            "disease, so pooling them hides the effect that matters."
        ),
        parser=p.parse_bmt_competing,
        tags=("clinical", "transplant", "competing-risks"),
        time_unit="days",
    ),
    "burn": DatasetSpec(
        name="burn",
        access=Access.OPEN,
        url=f"{_R}/KMsurv/burn.csv",
        sha256="8f6f55ef433dc43ed31b3c70df72447306336d77db257a86fe0f3806bf419551",
        license=_KMSURV_LICENSE,
        citation=(
            "Ichida, J. M. et al. (1993). Evaluation of protocol change in "
            "burn-care management using the Cox proportional hazards model "
            "with time-dependent covariates. Statistics in Medicine 12(3-4), "
            "301-310."
        ),
        description=(
            "154 severely burned patients randomised between routine "
            "bathing and a body-cleansing protocol, followed for "
            "staphylococcus aureus infection (48 infections). Covariates "
            "Z1-Z11 cover treatment, gender, race, burn percentage by site, "
            "and burn type. Excision and antibiotic administration are "
            "post-baseline treatment events and are dropped."
        ),
        parser=p.parse_burn,
        tags=("clinical", "trial"),
        time_unit="days",
    ),
    "bfeed": DatasetSpec(
        name="bfeed",
        access=Access.OPEN,
        url=f"{_R}/KMsurv/bfeed.csv",
        sha256="6f132ba187bc3e3c7c7b6bc1994b8e6afcaad196af4429a7f8071faf022ec203",
        license=_KMSURV_LICENSE,
        citation=_KM_BOOK,
        description=(
            "927 mothers from the National Survey of Family Growth, "
            "followed for weaning (892 events, 96%). Covariates: race, "
            "poverty status, smoking, alcohol use, age at birth, birth "
            "year, years of schooling, and prenatal care. Almost nothing is "
            "censored, which makes it a useful check that a model is not "
            "quietly depending on censoring being present."
        ),
        parser=p.parse_bfeed,
        tags=("behavioural", "survey"),
        time_unit="weeks",
    ),
    "std": DatasetSpec(
        name="std",
        access=Access.OPEN,
        url=f"{_R}/KMsurv/std.csv",
        sha256="7f463034f719ebf8ad94bcb3a91b3b212d43806aeaa7a50c53e2a8dfbfd85ea4",
        license=_KMSURV_LICENSE,
        citation=_KM_BOOK,
        description=(
            "877 patients treated for gonorrhea or chlamydia at an "
            "Indianapolis clinic, followed for reinfection (347 events). "
            "Twenty-odd covariates covering demographics, symptoms at "
            "presentation, condom use and partner count. One of the few "
            "cohorts here where the covariates are mostly binary symptom "
            "indicators."
        ),
        parser=p.parse_std,
        tags=("clinical", "epidemiology"),
        time_unit="days",
    ),
    "pneumon": DatasetSpec(
        name="pneumon",
        access=Access.OPEN,
        url=f"{_R}/KMsurv/pneumon.csv",
        sha256="c3109b3889d931f7f16f35815fae1ce5c7863230160a1002c78fb4968d3e56d1",
        license=_KMSURV_LICENSE,
        citation=_KM_BOOK,
        description=(
            "3470 infants from the National Longitudinal Survey of Youth, "
            "followed to first hospitalisation for pneumonia. Only 73 "
            "events -- a 2.1% event rate, the sparsest in the registry. "
            "Anything that needs a healthy number of events per covariate "
            "will struggle here, which is the reason to keep it."
        ),
        parser=p.parse_pneumon,
        tags=("clinical", "epidemiology", "rare-events"),
        time_unit="months",
    ),
    "hodgkins": DatasetSpec(
        name="hodgkins",
        access=Access.OPEN,
        url=f"{_R}/KMsurv/hodg.csv",
        sha256="3a8dc8b76289b77958daf33f19b8ecd629607eac231f0ae42d99d27edd9a2152",
        license=_KMSURV_LICENSE,
        citation=_KM_BOOK,
        description=(
            "43 lymphoma patients given allogeneic or autologous bone "
            "marrow transplants, crossed with Hodgkin's or non-Hodgkin's "
            "disease type, followed for death (26 events). Karnofsky score "
            "and waiting time to transplant are the covariates. A four-cell "
            "factorial design in a very small cohort."
        ),
        parser=p.parse_hodg,
        tags=("clinical", "transplant"),
        time_unit="days",
    ),
    "alloauto": DatasetSpec(
        name="alloauto",
        access=Access.OPEN,
        url=f"{_R}/KMsurv/alloauto.csv",
        sha256="f43f409a99670522c2d72e92d2c0ccffcff3b55feab53edcf023a1af69c5d99a",
        license=_KMSURV_LICENSE,
        citation=_KM_BOOK,
        description=(
            "101 leukemia patients given allogeneic (50) or autologous (51) "
            "transplants, followed for leukemia-free survival. The two "
            "survival curves cross, so the proportional hazards assumption "
            "fails visibly -- which is what the dataset is used to "
            "demonstrate, and why a log-rank test on it is misleading."
        ),
        parser=p.parse_alloauto,
        tags=("clinical", "transplant", "non-proportional"),
        time_unit="months",
    ),
    "btrial": DatasetSpec(
        name="btrial",
        access=Access.OPEN,
        url=f"{_R}/KMsurv/btrial.csv",
        sha256="b357969afc34ce3105ac647cb2c1d689d7a01323e9404fd93a8ab466ce741a43",
        license=_KMSURV_LICENSE,
        citation=_KM_BOOK,
        description=(
            "45 breast cancer patients split by immunohistochemical "
            "response (negative vs positive staining), followed for death "
            "(24 events). A two-sample comparison small enough to compute "
            "by hand."
        ),
        parser=p.parse_btrial,
        tags=("clinical",),
        time_unit="months",
    ),
    "bnct": DatasetSpec(
        name="bnct",
        access=Access.OPEN,
        url=f"{_R}/KMsurv/bnct.csv",
        sha256="d54e5081029222bd51e152e3b4d1de51474b94bdc50a168e5cd4e2cb2a88ccfe",
        license=_KMSURV_LICENSE,
        citation=_KM_BOOK,
        description=(
            "30 rats with implanted brain tumours, randomised between "
            "untreated, radiated, and radiated with boronophenylalanine, "
            "followed for death (27 events). A three-arm preclinical "
            "comparison with a clear dose-response ordering."
        ),
        parser=p.parse_bnct,
        tags=("preclinical", "trial"),
        time_unit="days",
    ),
    "drughiv": DatasetSpec(
        name="drughiv",
        access=Access.OPEN,
        url=f"{_R}/KMsurv/drughiv.csv",
        sha256="fbde90f735a42ac9645c79ea03d30ebcd36379565313a3b4c6d50f2c702e3ebc",
        license=_KMSURV_LICENSE,
        citation=_KM_BOOK,
        description=(
            "34 HIV-positive patients split by intravenous drug use "
            "history, followed for death (27 events). Klein and "
            "Moeschberger's two-sample exercise on tests weighted toward "
            "early versus late differences."
        ),
        parser=p.parse_drughiv,
        tags=("clinical",),
        time_unit="months",
    ),
    "psych": DatasetSpec(
        name="psych",
        access=Access.OPEN,
        url=f"{_R}/KMsurv/psych.csv",
        sha256="b21109063ee5514ed3cc2b3eff6c9503d6205c7ff6ee5f82dba6469a68302ffb",
        license=_KMSURV_LICENSE,
        citation=_KM_BOOK,
        description=(
            "26 psychiatric inpatients in Iowa followed from admission to "
            "death (14 events), with age at admission and sex. A "
            "left-truncation exercise on the age time scale, and small "
            "enough that the truncation's effect can be traced by hand."
        ),
        parser=p.parse_psych,
        tags=("clinical", "left-truncated"),
        time_unit="years",
    ),
    "twins": DatasetSpec(
        name="twins",
        access=Access.OPEN,
        url=f"{_R}/KMsurv/twins.csv",
        sha256="784362d17976dee8a85b76582c59751e0558d66109134cf5d4be9ee1a9326361",
        license=_KMSURV_LICENSE,
        citation=_KM_BOOK,
        description=(
            "24 Danish twins who survived past age 60, followed for death "
            "(8 events), recorded as matched pairs. The smallest paired "
            "design in the registry."
        ),
        parser=p.parse_twins,
        tags=("epidemiology", "clustered"),
        time_unit="years",
    ),
    "kidney_infection": DatasetSpec(
        name="kidney_infection",
        access=Access.OPEN,
        url=f"{_R}/KMsurv/kidney.csv",
        sha256="8ecbd76222eba9fb55c180c1ee5251063d3d14fb71adca70ab8d41b7f0713b80",
        license=_KMSURV_LICENSE,
        citation=(
            "Nahman, N. S. et al. (1992). Laparoscopic peritoneal dialysis "
            "catheter insertion. Advances in Peritoneal Dialysis 8, 404-407."
        ),
        description=(
            "119 kidney dialysis patients followed for exit-site infection, "
            "split by catheter placement technique (surgical vs "
            "percutaneous), 26 infections. Distinct from `kidney_catheter`, "
            "which is McGilchrist and Aisbett's recurrent-event study of a "
            "different cohort -- the two are routinely confused because R "
            "and KMsurv both call theirs `kidney`."
        ),
        parser=p.parse_kidney_infection,
        tags=("clinical",),
        time_unit="months",
    ),
    "allograft": DatasetSpec(
        name="allograft",
        access=Access.OPEN,
        url=f"{_R}/KMsurv/allograft.csv",
        sha256="2e5be60b1469b834afa2a1780e86058338873daa0e377955f855834f850b1e9e",
        license=_KMSURV_LICENSE,
        citation=_KM_BOOK,
        description=(
            "34 skin allografts on burn patients, followed to rejection (29 "
            "events), split by degree of HLA tissue match. Some patients "
            "received more than one graft, so rows cluster within patient."
        ),
        parser=p.parse_allograft,
        tags=("clinical", "transplant", "clustered"),
        time_unit="days",
    ),
    "baboon": DatasetSpec(
        name="baboon",
        access=Access.OPEN,
        url=f"{_R}/KMsurv/baboon.csv",
        sha256="3c2310cd73b755046d7d1ea7c50559b8bd535e1910a057570ca5fc7a7439878c",
        license=_KMSURV_LICENSE,
        citation=_KM_BOOK,
        description=(
            "152 observations of baboon troops descending from their "
            "sleeping trees in Kenya, with time of day as the time scale. "
            "Troops still in the tree when observation ended are censored, "
            "and troops already down when observers arrived are "
            "left-truncated. The one non-medical, non-industrial cohort "
            "here, and a good reminder that the time axis need not be "
            "duration since enrolment."
        ),
        parser=p.parse_baboon,
        tags=("ecology", "left-truncated"),
        time_unit="hours",
    ),
}
