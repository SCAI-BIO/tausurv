"""Registry entries for the deep-survival benchmark tables and gated cohorts.

The benchmark entries exist so a published number can be reproduced against
the table it was computed on. Every one of them is a *processed* table, and
each description says what the processing was -- the differences between
"METABRIC" as a molecular cohort and "METABRIC" as a nine-column benchmark
are large enough that treating them as one dataset produces nonsense.

The gated entries carry no data and never will. They are here because a
registry that quietly omits SEER and UNOS misrepresents what survival
analysis is practised on: the largest and most consequential cohorts in the
field are the ones you have to apply for.
"""

from __future__ import annotations

from tausurv.datasets import _parsers_bench as p
from tausurv.datasets._spec import Access, DatasetSpec

_SKSURV = (
    "https://raw.githubusercontent.com/sebp/scikit-survival/master/"
    "sksurv/datasets/data"
)
_DEEPSURV = (
    "https://raw.githubusercontent.com/jaredleekatzman/DeepSurv/master/"
    "experiments/data"
)
_HL_BOOK = (
    "Hosmer, D. W., Lemeshow, S. & May, S. (2008). Applied Survival "
    "Analysis: Regression Modeling of Time-to-Event Data, 2nd ed. Wiley."
)
_DEEPSURV_CITATION = (
    "Katzman, J. L. et al. (2018). DeepSurv: personalized treatment "
    "recommender system using a Cox proportional hazards deep neural "
    "network. BMC Medical Research Methodology 18, 24."
)

SPECS: dict[str, DatasetSpec] = {
    # ---- Hosmer & Lemeshow cohorts, via the scikit-survival ARFF bundle ----
    "whas500": DatasetSpec(
        name="whas500",
        access=Access.OPEN,
        url=f"{_SKSURV}/whas500.arff",
        sha256="f64040c91398875137822eca306a927208df6896808cd9764af7065722fa53d6",
        license="GPL-3.0 (redistributed in scikit-survival)",
        citation=_HL_BOOK,
        description=(
            "Worcester Heart Attack Study, a 500-patient sample followed "
            "from hospital admission for myocardial infarction until death "
            "(215 deaths, 43%). Fourteen covariates: age, gender, heart "
            "rate, blood pressures, BMI, and binary indicators for "
            "cardiovascular history, atrial fibrillation, cardiogenic "
            "shock, heart failure, AV block, MI order and MI type, plus "
            "length of stay. The reference dataset of Hosmer, Lemeshow and "
            "May's textbook, and a standard row in deep-survival benchmark "
            "tables."
        ),
        parser=p.parse_whas500,
        tags=("clinical", "benchmark", "cardiology"),
        time_unit="days",
    ),
    "aids": DatasetSpec(
        name="aids",
        access=Access.OPEN,
        url=f"{_SKSURV}/actg320.arff",
        sha256="f0613690853c36fc7b9b936cfae4c95ba460b64df1989cc1ccc126b729eae455",
        license="GPL-3.0 (redistributed in scikit-survival)",
        citation=(
            "Hammer, S. M. et al. (1997). A controlled trial of two "
            "nucleoside analogues plus indinavir in persons with human "
            "immunodeficiency virus infection and CD4 cell counts of 200 "
            "per cubic millimeter or less. New England Journal of Medicine "
            "337(11), 725-733."
        ),
        description=(
            "ACTG 320: 1151 HIV-positive patients randomised between a "
            "two-drug and a three-drug antiretroviral regimen, followed to "
            "an AIDS-defining event or death. Only 96 events (8.3%), which "
            "makes it the sparsest-event cohort in the registry and a real "
            "test of behaviour under heavy censoring. Eleven covariates "
            "including CD4 count, Karnofsky score, prior zidovudine "
            "exposure and treatment arm. `aids:death` uses the death-only "
            "endpoint; the unused endpoint's columns are dropped from both."
        ),
        parser=p.parse_actg320,
        tags=("clinical", "trial", "benchmark", "rare-events"),
        time_unit="days",
    ),
    "aids:death": DatasetSpec(
        name="aids:death",
        access=Access.OPEN,
        url=f"{_SKSURV}/actg320.arff",
        sha256="f0613690853c36fc7b9b936cfae4c95ba460b64df1989cc1ccc126b729eae455",
        license="GPL-3.0 (redistributed in scikit-survival)",
        citation=(
            "Hammer, S. M. et al. (1997). A controlled trial of two "
            "nucleoside analogues plus indinavir in persons with human "
            "immunodeficiency virus infection and CD4 cell counts of 200 "
            "per cubic millimeter or less. New England Journal of Medicine "
            "337(11), 725-733."
        ),
        description=(
            "ACTG 320 with death alone as the endpoint, rather than the "
            "trial's composite of AIDS-defining event or death. Rarer "
            "still than the composite, and the pair makes a clean "
            "illustration of how much a composite endpoint buys in "
            "statistical power."
        ),
        parser=p.parse_actg320_death,
        tags=("clinical", "trial", "rare-events"),
        time_unit="days",
    ),
    "breast_cancer_gse7390": DatasetSpec(
        name="breast_cancer_gse7390",
        access=Access.OPEN,
        url=f"{_SKSURV}/breast_cancer_GSE7390-metastasis.arff",
        sha256="233f4c1c06b2ee299fffcba0f9881f6ed66a3566ccbec30b5004f0d298b52570",
        license="GPL-3.0 (redistributed in scikit-survival)",
        citation=(
            "Desmedt, C. et al. (2007). Strong time dependence of the "
            "76-gene prognostic signature for node-negative breast cancer "
            "patients in the TRANSBIG multicenter independent validation "
            "series. Clinical Cancer Research 13(11), 3207-3214."
        ),
        description=(
            "198 node-negative breast cancer patients with gene expression "
            "measured on the Affymetrix U133A array, followed for distant "
            "metastasis. Alongside age, tumour size, grade and ER status "
            "sit the 76 probe sets of the Desmedt prognostic signature, "
            "which makes this the widest cohort in the registry and the "
            "only one where covariates approach the sample size."
        ),
        parser=p.parse_gse7390,
        tags=("clinical", "genomics", "high-dimensional"),
        time_unit="years",
    ),
    # ---- DeepSurv benchmark tables (HDF5; needs the `deepsurv` extra) ----
    "metabric": DatasetSpec(
        name="metabric",
        access=Access.OPEN,
        url=f"{_DEEPSURV}/metabric/metabric_IHC4_clinical_train_test.h5",
        sha256="e959131973c906c7e8b1e662c81160914768156a6da145a9bfed9b9ee8bf3e72",
        license="MIT (redistributed in DeepSurv)",
        citation=(
            "Curtis, C. et al. (2012). The genomic and transcriptomic "
            "architecture of 2,000 breast tumours reveals novel subgroups. "
            "Nature 486(7403), 346-352."
        ),
        description=(
            "METABRIC as processed for DeepSurv: 1904 breast cancer "
            "patients, 1103 deaths (57.9%), nine covariates. This "
            "nine-column table is what deep-survival papers mean by "
            "METABRIC; the full molecular cohort with expression and "
            "copy-number data is `metabric:full` and needs a cBioPortal "
            "download. Katzman et al. describe the covariates as four gene "
            "indicators (MKI67, EGFR, PGR, ERBB2) plus hormone treatment, "
            "radiotherapy, chemotherapy, ER status and age at diagnosis, "
            "but the HDF5 file stores no column names, so columns are "
            "exposed positionally as `x0`..`x8` and that ordering is not "
            "verifiable from the data. DeepSurv's train/test split is kept "
            "as a `split` column."
        ),
        parser=p.parse_metabric,
        tags=("clinical", "benchmark", "breast-cancer"),
        time_unit="months",
        requires="h5py",
    ),
    "gbsg_rotterdam": DatasetSpec(
        name="gbsg_rotterdam",
        access=Access.OPEN,
        url=f"{_DEEPSURV}/gbsg/gbsg_cancer_train_test.h5",
        sha256="bd455f4395322a8b23040c15cedbef00ad51a447d02c95f1f1d686a6b196484b",
        license="MIT (redistributed in DeepSurv)",
        citation=_DEEPSURV_CITATION,
        description=(
            "The DeepSurv GBSG benchmark: 2232 patients, being the 1546 "
            "Rotterdam tumour bank patients as training data and the 686 "
            "German Breast Cancer Study Group trial patients as test data, "
            "combined into one table with seven covariates. This is a "
            "different cohort from `gbsg`, which is the German trial alone. "
            "It is deliberately not called `gbsg:deepsurv`: a variant "
            "suffix means another reading of the same rows, and this "
            "table combines two studies. Both spellings resolve. The "
            "name collision is real in the literature -- comparing a "
            "number computed on one against a number computed on the other "
            "is comparing 686 patients to 2232. For the same endpoint on "
            "the Rotterdam side alone, see `rotterdam:rfs`. Columns are "
            "positional (`x0`..`x6`): the HDF5 file stores no names. The "
            "original split is kept as a `split` column."
        ),
        parser=p.parse_gbsg_deepsurv,
        tags=("clinical", "benchmark", "breast-cancer"),
        time_unit="days",
        requires="h5py",
    ),
    "support:deepsurv": DatasetSpec(
        name="support:deepsurv",
        access=Access.OPEN,
        url=f"{_DEEPSURV}/support/support_train_test.h5",
        sha256="e398f49930e4efc4ecd675a72ac547f0d1ae25ccf13f6a1024cded12ea88a83e",
        license="MIT (redistributed in DeepSurv)",
        citation=(
            "Knaus, W. A. et al. (1995). The SUPPORT prognostic model: "
            "objective estimates of survival for seriously ill hospitalized "
            "adults. Annals of Internal Medicine 122(3), 191-203."
        ),
        description=(
            "SUPPORT as processed for DeepSurv: 8873 patients and 14 "
            "baseline clinical covariates, selected from the 47 columns of "
            "the raw cohort. The selection excludes every post-baseline and "
            "model-derived column, so benchmark numbers on this table are "
            "lower than numbers on the raw `support` entry, and more "
            "meaningful. Compare against `support:nonleaky`, which applies "
            "the same principle to the full 9105-patient cohort. Columns "
            "are positional (`x0`..`x13`): the HDF5 file stores no names."
        ),
        parser=p.parse_support_deepsurv,
        tags=("clinical", "benchmark", "critical-care"),
        time_unit="days",
        requires="h5py",
    ),
    "whas_deepsurv": DatasetSpec(
        name="whas_deepsurv",
        access=Access.OPEN,
        url=f"{_DEEPSURV}/whas/whas_train_test.h5",
        sha256="2ed86f073fdbe34be10f7e485eeaf67b61004521390c46b9a18aa9b82d5e8f9c",
        license="MIT (redistributed in DeepSurv)",
        citation=_HL_BOOK,
        description=(
            "The Worcester Heart Attack Study as processed for DeepSurv: "
            "1638 patients and six covariates. A larger slice of the study "
            "than the 500-patient `whas500` sample, with a much reduced "
            "covariate set -- the two are not substitutes. Katzman et al. "
            "describe five variables (age, sex, BMI, heart failure, MI "
            "order) while the file ships six, which is the clearest reason "
            "columns here are exposed positionally as `x0`..`x5` rather "
            "than under names that cannot be checked."
        ),
        parser=p.parse_whas_deepsurv,
        tags=("clinical", "benchmark", "cardiology"),
        time_unit="days",
        requires="h5py",
    ),
    # ---- gated: specs and parsers, no data ----
    "seer": DatasetSpec(
        name="seer",
        access=Access.USER_PROVIDED,
        license="NCI SEER Research Data Use Agreement (no redistribution)",
        citation=(
            "Surveillance, Epidemiology, and End Results (SEER) Program, "
            "National Cancer Institute. SEER Research Data."
        ),
        description=(
            "The SEER cancer registry: population-based incidence and "
            "survival for roughly half the United States, and the largest "
            "cohort survival analysis is routinely practised on -- "
            "published SEER survival studies typically use 10^5 to 10^6 "
            "patients. Cause of death distinguishes cancer-specific from "
            "other mortality, so it is naturally competing-risks. The "
            "parser expects a SEER*Stat CSV export and resolves the "
            "survival-time and cause-of-death columns case-insensitively, "
            "since their names vary by SEER*Stat version. For a "
            "redistributable SEER-derived stand-in, see `prostate`."
        ),
        access_help=(
            "1. Request access at https://seer.cancer.gov/data/access.html\n"
            "2. Sign the SEER Research Data Use Agreement\n"
            "3. Install SEER*Stat and export a case-listing session to CSV,\n"
            "   including at minimum: Survival months, COD to site recode,\n"
            "   and whatever covariates you need\n"
            "4. load_dataset('seer', path='/dir/containing/the/export')"
        ),
        parser=p.parse_seer,
        tags=("clinical", "competing-risks", "registry", "large"),
        time_unit="months",
    ),
    "unos": DatasetSpec(
        name="unos",
        access=Access.USER_PROVIDED,
        license="OPTN/UNOS Data Use Agreement (no redistribution)",
        citation=(
            "Organ Procurement and Transplantation Network (OPTN) / United "
            "Network for Organ Sharing (UNOS). Standard Transplant Analysis "
            "and Research (STAR) files."
        ),
        description=(
            "The OPTN/UNOS national transplant registry: every solid-organ "
            "transplant performed in the United States, with post-transplant "
            "graft and patient survival. Around 60000 records per organ "
            "cohort with several dozen donor and recipient covariates. The "
            "parser reads a STAR file as delimited text and uses PTIME / "
            "PSTATUS (patient survival); the graft-survival columns are "
            "dropped so they cannot leak."
        ),
        access_help=(
            "1. Submit a data request at https://optn.transplant.hrsa.gov/"
            "data/request-data/\n"
            "2. Execute the OPTN Data Use Agreement\n"
            "3. Extract the STAR file archive you receive\n"
            "4. load_dataset('unos', path='/dir/containing/the/STAR/files')"
        ),
        parser=p.parse_unos,
        tags=("clinical", "transplant", "registry", "large"),
        time_unit="days",
    ),
    "metabric:full": DatasetSpec(
        name="metabric:full",
        access=Access.CREDENTIALED,
        license="METABRIC / cBioPortal terms; EGA controlled access for raw data",
        citation=(
            "Curtis, C. et al. (2012). The genomic and transcriptomic "
            "architecture of 2,000 breast tumours reveals novel subgroups. "
            "Nature 486(7403), 346-352."
        ),
        description=(
            "The full METABRIC cohort: ~1980 breast tumours with clinical "
            "annotation, expression and copy-number data, rather than the "
            "nine columns of the `metabric` benchmark table. Overall "
            "survival in months is the endpoint. Use this when the question "
            "is about high-dimensional molecular prediction; use "
            "`metabric` when the question is reproducing a published "
            "deep-survival number."
        ),
        access_help=(
            "1. Download the brca_metabric study from "
            "https://www.cbioportal.org/study/summary?id=brca_metabric\n"
            "   (the 'Download' tab gives a tar.gz of the whole study)\n"
            "2. Extract it\n"
            "3. load_dataset('metabric:full', path='/path/to/brca_metabric')\n"
            "\nRaw sequencing data requires separate EGA controlled access."
        ),
        parser=p.parse_metabric_full,
        tags=("clinical", "genomics", "breast-cancer", "high-dimensional"),
        time_unit="months",
    ),
    "mimic": DatasetSpec(
        name="mimic",
        access=Access.CREDENTIALED,
        license="PhysioNet Credentialed Health Data License 1.5.0",
        citation=(
            "Johnson, A. E. W. et al. (2023). MIMIC-IV, a freely accessible "
            "electronic health record dataset. Scientific Data 10, 1."
        ),
        description=(
            "MIMIC-IV: intensive care and hospital records for around "
            "300000 patients at Beth Israel Deaconess Medical Center. The "
            "parser builds a mortality cohort from the `hosp` module's "
            "patient table. Dates are de-identified by a per-patient shift, "
            "so absolute times are not comparable across patients -- the "
            "anchor age and follow-up window are what the endpoint is built "
            "from, and any serious use should derive its own cohort."
        ),
        access_help=(
            "1. Complete CITI 'Data or Specimens Only Research' training\n"
            "2. Apply for credentialed access at "
            "https://physionet.org/settings/credentialing/\n"
            "3. Sign the data use agreement for MIMIC-IV\n"
            "4. Download the `hosp` module\n"
            "5. load_dataset('mimic', path='/path/to/mimiciv/hosp')"
        ),
        parser=p.parse_mimic,
        tags=("clinical", "critical-care", "ehr", "large"),
        time_unit="years",
    ),
    "framingham": DatasetSpec(
        name="framingham",
        access=Access.CREDENTIALED,
        license="NHLBI BioLINCC Data Use Agreement",
        citation=(
            "Dawber, T. R., Meadors, G. F. & Moore, F. E. (1951). "
            "Epidemiological approaches to heart disease: the Framingham "
            "Study. American Journal of Public Health 41(3), 279-286."
        ),
        description=(
            "The Framingham Heart Study teaching dataset (`frmgham2`): 4434 "
            "participants followed for up to 24 years, with mortality and "
            "cardiovascular endpoints. The longitudinal file holds up to "
            "three examination periods per subject; the parser keeps period "
            "1 so each row is one participant at baseline, and drops the "
            "non-mortality endpoint columns so they cannot leak. The "
            "teaching dataset is a simplified extract, not the full study."
        ),
        access_help=(
            "1. Request the Framingham teaching dataset at "
            "https://biolincc.nhlbi.nih.gov/teaching/\n"
            "2. Accept the BioLINCC data use terms\n"
            "3. Extract the archive (it contains frmgham2.csv)\n"
            "4. load_dataset('framingham', path='/dir/with/frmgham2.csv')"
        ),
        parser=p.parse_framingham,
        tags=("clinical", "epidemiology", "cardiology", "longitudinal"),
        time_unit="days",
    ),
}
