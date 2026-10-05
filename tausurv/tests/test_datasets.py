from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from tausurv.datasets import (
    Access,
    DatasetIntegrityError,
    OfflineModeError,
    SurvivalBunch,
    UnknownDatasetError,
    dataset_info,
    list_datasets,
    load_capacitor,
    load_colon,
    load_dataset,
    load_flchain,
    load_gbsg,
    load_genfan,
    load_ifluid,
    load_imotor,
    load_kidney_transplant,
    load_larynx,
    load_lung,
    load_melanoma,
    load_mgus2,
    load_nwtco,
    load_pbc,
    load_rossi,
    load_support,
    load_telco_churn,
    load_tongue,
    load_veteran,
    load_waltons,
)
from tausurv.datasets._cache import cached_path, fetch_to_cache, resolve_cache_dir
from tausurv.datasets._registry import resolve_name


#: The twenty cohorts the registry shipped with. Later additions must not
#: rename or drop any of them: a name in this list is a promise, because
#: results are reported against it.
V1_NAMES = (
    "capacitor",
    "colon",
    "flchain",
    "gbsg",
    "genfan",
    "ifluid",
    "imotor",
    "kidney_transplant",
    "larynx",
    "lung",
    "melanoma",
    "mgus2",
    "nwtco",
    "pbc",
    "rossi",
    "support",
    "telco_churn",
    "tongue",
    "veteran",
    "waltons",
)


def test_v1_names_are_all_still_registered():
    registered = set(list_datasets())
    assert set(V1_NAMES) <= registered


def test_list_datasets_is_sorted_and_free_of_aliases():
    names = list_datasets()
    assert names == sorted(names)
    assert len(names) == len(set(names))
    # An alias must resolve but never appear: one canonical name per table.
    assert "gbsg2" not in names
    assert resolve_name("gbsg2") == "gbsg"
    assert "actg320" not in names
    assert resolve_name("actg320") == "aids"


def test_base_only_drops_variants_but_keeps_their_study():
    every, studies = list_datasets(), list_datasets(base_only=True)
    assert "pbc:randomised" in every
    assert "pbc:randomised" not in studies
    assert "pbc" in studies
    assert set(studies) <= set(every)


def test_every_variant_declares_a_registered_base():
    """`base:variant` promises another reading of `base`, so `base` must exist.

    A table that merely shares a name with another study is not a variant of
    it -- `gbsg_rotterdam` combines two cohorts and is deliberately not
    called `gbsg:deepsurv`, because that would claim it is 686 patients
    read differently when it is 2232 patients from two trials.
    """
    names = set(list_datasets())
    for name in names:
        base, sep, _ = name.partition(":")
        if sep:
            assert base in names, f"{name} has no registered base {base!r}"


def test_variants_of_one_source_file_share_its_digest():
    """Variants built from a base's file must not drift onto other bytes."""
    by_url: dict[str, set[str]] = {}
    for name in list_datasets(access=Access.OPEN):
        info = dataset_info(name)
        assert info.url is not None
        by_url.setdefault(info.url, set()).add(info.sha256 or "")
    for url, digests in by_url.items():
        assert len(digests) == 1, f"{url} pinned to several digests: {digests}"


def test_every_open_dataset_is_pinned():
    for name in list_datasets(access=Access.OPEN):
        info = dataset_info(name)
        assert info.url, name
        assert info.sha256 and len(info.sha256) == 64, name


def test_gated_datasets_carry_setup_instructions():
    """A dataset nobody can download is useless without the way to get it."""
    from tausurv.datasets._registry import _REGISTRY

    gated = [s for s in _REGISTRY.values() if s.access is not Access.OPEN]
    assert gated, "the gated tier should not be empty"
    for spec in gated:
        assert spec.url is None and spec.sha256 is None, spec.name
        assert spec.access_help and "load_dataset" in spec.access_help, spec.name


def test_unknown_name_suggests_variants_of_a_known_base():
    with pytest.raises(UnknownDatasetError) as excinfo:
        load_dataset("pbc:nonesuch")
    assert "pbc:randomised" in str(excinfo.value)


def test_list_datasets_filter_by_tag():
    assert {"capacitor", "genfan", "ifluid", "imotor"} <= set(
        list_datasets(tag="reliability")
    )
    assert list_datasets(tag="churn") == ["telco_churn"]
    for name in ("mgus2", "melanoma", "colon"):
        assert name in list_datasets(tag="competing-risks")
    assert "pbc" not in list_datasets(tag="competing-risks")
    assert "rossi" in list_datasets(tag="recidivism")
    assert list_datasets(tag="does-not-exist") == []


def test_dataset_info_carries_tags_and_time_unit(monkeypatch, tmp_path):
    monkeypatch.setenv("TAUSURV_DATA", str(tmp_path))
    monkeypatch.setenv("TAUSURV_OFFLINE", "1")
    pbc = dataset_info("pbc")
    assert pbc.tags == ("clinical",)
    assert pbc.time_unit == "days"
    mgus2 = dataset_info("mgus2")
    assert "competing-risks" in mgus2.tags
    assert mgus2.time_unit == "months"
    genfan = dataset_info("genfan")
    assert genfan.tags == ("reliability",)
    assert genfan.time_unit == "hours"


def test_survival_bunch_tuple_unpacks():
    bunch = SurvivalBunch(
        X=pl.DataFrame({"a": [1, 2, 3]}),
        event_time=np.array([1.0, 2.0, 3.0]),
        event_indicator=np.array([1, 0, 1], dtype=np.int8),
        feature_names=("a",),
        name="toy",
        description="",
        citation="",
        license="",
        url="",
    )
    X, Y, delta = bunch
    assert X is bunch.X
    assert Y is bunch.event_time
    assert delta is bunch.event_indicator


def test_survival_bunch_single_event_defaults():
    bunch = SurvivalBunch(
        X=pl.DataFrame({"a": [1, 2, 3]}),
        event_time=np.array([1.0, 2.0, 3.0]),
        event_indicator=np.array([1, 0, 1], dtype=np.int8),
        feature_names=("a",),
        name="toy",
        description="",
        citation="",
        license="",
        url="",
    )
    assert bunch.cause is None
    assert bunch.n_causes == 1
    assert bunch.cause_labels is None
    assert bunch.is_competing_risks is False


def test_survival_bunch_competing_risks_shape():
    cause = np.array([0, 1, 2, 1, 0], dtype=np.int8)
    bunch = SurvivalBunch(
        X=pl.DataFrame({"a": [1, 2, 3, 4, 5]}),
        event_time=np.array([10.0, 5.0, 7.0, 3.0, 12.0]),
        event_indicator=(cause > 0).astype(np.int8),
        feature_names=("a",),
        name="toy_cr",
        description="",
        citation="",
        license="",
        url="",
        cause=cause,
        n_causes=2,
        cause_labels=("a", "b"),
    )
    assert bunch.is_competing_risks is True
    assert bunch.n_causes == 2
    assert bunch.cause_labels == ("a", "b")
    # Tuple-unpack still yields event_indicator (binary), not cause.
    _, _, delta = bunch
    assert delta.dtype == np.int8
    assert int(delta.sum()) == 3
    # Cause vector accessed via attribute.
    np.testing.assert_array_equal(bunch.cause, [0, 1, 2, 1, 0])


def test_dataset_info_does_not_download(tmp_path, monkeypatch):
    monkeypatch.setenv("TAUSURV_DATA", str(tmp_path))
    monkeypatch.setenv("TAUSURV_OFFLINE", "1")
    info = dataset_info("pbc")
    assert info.name == "pbc"
    assert info.access is Access.OPEN
    assert info.url is not None and info.sha256 is not None
    assert "Therneau" in info.citation
    assert not any(tmp_path.iterdir())


def test_unknown_dataset_raises_with_available_list():
    with pytest.raises(UnknownDatasetError, match="unknown dataset"):
        load_dataset("does_not_exist")


def test_resolve_cache_dir_priority(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    via_env = tmp_path / "env"
    monkeypatch.setenv("TAUSURV_DATA", str(via_env))
    assert resolve_cache_dir(explicit) == explicit.resolve()
    assert resolve_cache_dir(None) == via_env.resolve()
    monkeypatch.delenv("TAUSURV_DATA")
    assert isinstance(resolve_cache_dir(None), Path)


def test_cached_path_uses_url_basename(tmp_path, monkeypatch):
    monkeypatch.setenv("TAUSURV_DATA", str(tmp_path))
    p = cached_path("foo", "https://example.com/some/dir/foo.csv")
    assert p == (tmp_path / "foo" / "foo.csv").resolve()


class _FakeResponse:
    """Minimal stand-in for urlopen()'s returned context manager."""

    def __init__(self, body: bytes) -> None:
        self.headers = {"Content-Length": str(len(body))}
        self._left = body

    def read(self, n: int = -1) -> bytes:
        chunk, self._left = self._left, b""
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_fetch_to_cache_writes_file_and_metadata(tmp_path, monkeypatch):
    body = b"col_a,col_b\n1,2\n"
    sha = hashlib.sha256(body).hexdigest()
    monkeypatch.setenv("TAUSURV_DATA", str(tmp_path))
    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
        path = fetch_to_cache("toy", "https://example.com/toy.csv", sha)
    assert path.exists() and path.read_bytes() == body
    meta = json.loads((path.parent / "metadata.json").read_text())
    assert meta["sha256"] == sha
    assert meta["name"] == "toy"
    assert meta["filename"] == "toy.csv"


def test_fetch_to_cache_integrity_failure_removes_partial(tmp_path, monkeypatch):
    body = b"hello\n"
    monkeypatch.setenv("TAUSURV_DATA", str(tmp_path))
    with (
        patch("urllib.request.urlopen", return_value=_FakeResponse(body)),
        pytest.raises(DatasetIntegrityError, match="sha256 mismatch"),
    ):
        fetch_to_cache("fake", "https://example.com/x.csv", "0" * 64)
    assert not (tmp_path / "fake" / "x.csv.part").exists()
    assert not (tmp_path / "fake" / "x.csv").exists()


def test_fetch_to_cache_skips_when_already_cached(tmp_path, monkeypatch):
    body = b"a,b\n1,2\n"
    sha = hashlib.sha256(body).hexdigest()
    monkeypatch.setenv("TAUSURV_DATA", str(tmp_path))
    with patch("urllib.request.urlopen") as opener:
        opener.side_effect = lambda *a, **k: _FakeResponse(body)
        first = fetch_to_cache("toy", "https://example.com/toy.csv", sha)
        second = fetch_to_cache("toy", "https://example.com/toy.csv", sha)
    assert first == second
    assert opener.call_count == 1


def test_fetch_to_cache_offline_mode_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("TAUSURV_DATA", str(tmp_path))
    monkeypatch.setenv("TAUSURV_OFFLINE", "1")
    with pytest.raises(OfflineModeError, match="not cached"):
        fetch_to_cache("toy", "https://example.com/toy.csv", "0" * 64)


@pytest.mark.network
def test_load_pbc_end_to_end(network_cache):
    ds = load_pbc(cache_dir=network_cache)
    assert isinstance(ds, SurvivalBunch)
    assert isinstance(ds.X, pl.DataFrame)
    assert ds.n == 418
    assert ds.d == 17
    assert ds.event_time.dtype == np.float64
    assert ds.event_indicator.dtype == np.int8
    assert int(ds.event_indicator.sum()) == 161
    assert ds.feature_names == tuple(ds.X.columns)
    assert "trt" in ds.feature_names and "sex" in ds.feature_names


@pytest.mark.network
def test_load_rossi_end_to_end(network_cache):
    ds = load_rossi(cache_dir=network_cache)
    assert isinstance(ds.X, pl.DataFrame)
    assert ds.n == 432
    assert ds.d == 7
    assert int(ds.event_indicator.sum()) == 114
    assert ds.feature_names == ("fin", "age", "race", "wexp", "mar", "paro", "prio")


@pytest.mark.network
def test_load_gbsg_end_to_end(network_cache):
    ds = load_gbsg(cache_dir=network_cache)
    assert isinstance(ds.X, pl.DataFrame)
    assert ds.n == 686
    assert ds.d == 8
    assert int(ds.event_indicator.sum()) == 299


@pytest.mark.network
def test_load_genfan_reliability(network_cache):
    ds = load_genfan(cache_dir=network_cache)
    assert isinstance(ds.X, pl.DataFrame)
    assert ds.n == 70
    assert ds.d == 0  # genfan has no covariates besides the outcome
    assert int(ds.event_indicator.sum()) == 12
    assert ds.tags == ("reliability",)
    assert ds.time_unit == "hours"
    assert ds.is_competing_risks is False


@pytest.mark.network
def test_load_ifluid_all_events(network_cache):
    """ifluid has no status column; loader synthesises all-event indicator."""
    ds = load_ifluid(cache_dir=network_cache)
    assert ds.n == 41
    assert ds.d == 1
    assert ds.feature_names == ("voltage",)
    assert int(ds.event_indicator.sum()) == 41  # all events
    assert ds.time_unit == "minutes"
    assert ds.tags == ("reliability",)


@pytest.mark.network
def test_load_imotor_reliability(network_cache):
    ds = load_imotor(cache_dir=network_cache)
    assert ds.n == 40
    assert ds.d == 1
    assert ds.feature_names == ("temp",)
    assert int(ds.event_indicator.sum()) == 17
    assert ds.time_unit == "hours"
    assert ds.tags == ("reliability",)


@pytest.mark.network
def test_load_capacitor_reliability(network_cache):
    ds = load_capacitor(cache_dir=network_cache)
    assert ds.n == 64
    assert ds.d == 2
    assert "temperature" in ds.feature_names
    assert "voltage" in ds.feature_names
    assert "fail" not in ds.feature_names  # Type-II artifact dropped
    assert int(ds.event_indicator.sum()) == 32
    assert ds.time_unit == "hours"
    assert ds.tags == ("reliability",)


@pytest.mark.network
def test_load_lung_end_to_end(network_cache):
    ds = load_lung(cache_dir=network_cache)
    assert isinstance(ds.X, pl.DataFrame)
    assert ds.n == 228
    assert ds.d == 8
    assert int(ds.event_indicator.sum()) == 165
    assert "ph.ecog" in ds.feature_names


@pytest.mark.network
def test_load_veteran_end_to_end(network_cache):
    ds = load_veteran(cache_dir=network_cache)
    assert isinstance(ds.X, pl.DataFrame)
    assert ds.n == 137
    assert ds.d == 6
    assert int(ds.event_indicator.sum()) == 128
    assert "celltype" in ds.feature_names


@pytest.mark.network
def test_load_flchain_end_to_end(network_cache):
    ds = load_flchain(cache_dir=network_cache)
    assert isinstance(ds.X, pl.DataFrame)
    assert ds.n == 7874
    assert ds.d == 8  # chapter dropped
    assert int(ds.event_indicator.sum()) == 2169
    assert "chapter" not in ds.feature_names


@pytest.mark.network
def test_load_support_end_to_end(network_cache):
    ds = load_support(cache_dir=network_cache)
    assert isinstance(ds.X, pl.DataFrame)
    assert ds.n == 9105
    assert ds.d == 45  # 48 cols - row_id - d.time - death
    assert int(ds.event_indicator.sum()) == 6201
    assert "age" in ds.feature_names
    assert "dzgroup" in ds.feature_names
    assert "d.time" not in ds.feature_names
    # Numeric-as-string columns are coerced to Float64.
    assert ds.X.schema["age"] == pl.Float64
    # Genuine categoricals stay as strings.
    assert ds.X.schema["dzgroup"] == pl.String


@pytest.mark.network
def test_load_melanoma_competing_risks(network_cache):
    ds = load_melanoma(cache_dir=network_cache)
    assert isinstance(ds.X, pl.DataFrame)
    assert ds.n == 205
    assert ds.d == 5  # sex, age, year, thickness, ulcer
    assert ds.is_competing_risks is True
    assert ds.n_causes == 2
    assert ds.cause_labels == ("melanoma death", "death from other causes")
    values, counts = np.unique(ds.cause, return_counts=True)
    assert dict(zip(values.tolist(), counts.tolist(), strict=True)) == {
        0: 134,
        1: 57,
        2: 14,
    }
    assert int(ds.event_indicator.sum()) == 71  # 57 + 14


@pytest.mark.network
def test_load_mgus2_competing_risks(network_cache):
    ds = load_mgus2(cache_dir=network_cache)
    assert isinstance(ds.X, pl.DataFrame)
    assert ds.n == 1384
    assert ds.d == 6  # age, sex, dxyr, hgb, creat, mspike
    assert ds.is_competing_risks is True
    assert ds.n_causes == 2
    assert ds.cause_labels == ("plasma cell malignancy", "death")
    # First-event-wins recoding produces 409 censored, 115 PCM, 860 death.
    values, counts = np.unique(ds.cause, return_counts=True)
    assert dict(zip(values.tolist(), counts.tolist(), strict=True)) == {
        0: 409,
        1: 115,
        2: 860,
    }
    # event_indicator is derived as (cause > 0); 115 + 860 = 975.
    assert int(ds.event_indicator.sum()) == 975
    # event_time = ptime when PCM first, futime otherwise.
    assert ds.event_time.dtype == np.float64


@pytest.mark.network
def test_load_pbc_is_not_competing_risks(network_cache):
    """Sanity: single-event loaders leave cause/n_causes at defaults."""
    ds = load_pbc(cache_dir=network_cache)
    assert ds.is_competing_risks is False
    assert ds.cause is None
    assert ds.n_causes == 1
    assert ds.cause_labels is None


@pytest.mark.network
def test_load_nwtco_end_to_end(network_cache):
    ds = load_nwtco(cache_dir=network_cache)
    assert ds.n == 4028
    assert int(ds.event_indicator.sum()) == 571
    assert "in.subcohort" in ds.feature_names
    assert ds.tags == ("clinical",)


@pytest.mark.network
def test_load_colon_competing_risks(network_cache):
    ds = load_colon(cache_dir=network_cache)
    assert ds.n == 929
    assert ds.is_competing_risks is True
    assert ds.cause_labels == ("recurrence", "death")
    values, counts = np.unique(ds.cause, return_counts=True)
    assert dict(zip(values.tolist(), counts.tolist(), strict=True)) == {
        0: 423,
        1: 468,
        2: 38,
    }


@pytest.mark.network
def test_load_larynx_end_to_end(network_cache):
    ds = load_larynx(cache_dir=network_cache)
    assert ds.n == 90
    assert int(ds.event_indicator.sum()) == 50
    assert ds.time_unit == "months"


@pytest.mark.network
def test_load_tongue_end_to_end(network_cache):
    ds = load_tongue(cache_dir=network_cache)
    assert ds.n == 80
    assert int(ds.event_indicator.sum()) == 53
    assert ds.time_unit == "weeks"
    assert ds.feature_names == ("type",)


@pytest.mark.network
def test_load_kidney_transplant_end_to_end(network_cache):
    ds = load_kidney_transplant(cache_dir=network_cache)
    assert ds.n == 863
    assert int(ds.event_indicator.sum()) == 140
    assert "black_male" in ds.feature_names


@pytest.mark.network
def test_load_waltons_end_to_end(network_cache):
    ds = load_waltons(cache_dir=network_cache)
    assert ds.n == 163
    assert int(ds.event_indicator.sum()) == 156
    assert ds.feature_names == ("group",)


@pytest.mark.network
def test_load_telco_churn_end_to_end(network_cache):
    ds = load_telco_churn(cache_dir=network_cache)
    assert ds.n == 7043
    assert int(ds.event_indicator.sum()) == 1869
    assert ds.tags == ("churn",)
    assert ds.time_unit == "months"
    assert "customerID" not in ds.feature_names


@pytest.mark.network
def test_load_dataset_cache_hits_on_repeat(network_cache):
    a = load_pbc(cache_dir=network_cache)
    b = load_dataset("pbc", cache_dir=network_cache)
    assert a.n == b.n
    # The cache is content-addressed: <root>/<sha256[:16]>/<filename>.
    assert cached_path(
        dataset_info("pbc").sha256, dataset_info("pbc").url, network_cache
    ).exists()


@pytest.mark.network
def test_variants_of_one_study_share_a_single_download(network_cache):
    """Three readings of PBC must not cost three copies of the same CSV."""
    paths = set()
    for name in ("pbc", "pbc:randomised", "pbc:transplant"):
        load_dataset(name, cache_dir=network_cache)
        info = dataset_info(name)
        assert info.url is not None and info.sha256 is not None
        paths.add(cached_path(info.sha256, info.url, network_cache))
    assert len(paths) == 1, f"one CSV expected, got {sorted(paths)}"
    assert next(iter(paths)).exists()


@pytest.mark.network
@pytest.mark.parametrize("name", list_datasets(access=Access.OPEN))
def test_every_open_dataset_loads_and_is_well_formed(name, network_cache):
    """Load every open dataset and check the contract each one promises.

    This is the test that stops a registry of this size from rotting: a
    parser that silently reads the wrong column, or an upstream file that
    changes shape, shows up here rather than in someone's results.
    """
    if (requires := dataset_info(name).requires) is not None:
        pytest.importorskip(requires)
    ds = load_dataset(name, cache_dir=network_cache)

    assert ds.n > 0
    assert ds.event_time.shape == (ds.n,)
    assert ds.event_indicator.shape == (ds.n,)
    assert np.isfinite(ds.event_time).all(), "non-finite event times"
    assert (ds.event_time >= 0).all(), "negative event times"
    assert set(np.unique(ds.event_indicator)) <= {0, 1}
    assert ds.event_indicator.sum() > 0, "no events at all"
    assert ds.feature_names == tuple(ds.X.columns)
    if ds.d:
        assert ds.X.height == ds.n
    if ds.is_competing_risks:
        assert ds.cause is not None
        assert set(np.unique(ds.cause)) <= set(range(ds.n_causes + 1))
        # event_indicator is the derived any-cause view of cause.
        assert np.array_equal(ds.event_indicator, (ds.cause > 0).astype(np.int8))
        if ds.cause_labels is not None:
            assert len(ds.cause_labels) == ds.n_causes


@pytest.mark.network
def test_endpoint_columns_never_survive_into_covariates(network_cache):
    """A covariate named like the endpoint is the classic silent leak."""
    banned = {
        "time",
        "status",
        "delta",
        "event",
        "death",
        "futime",
        "fustat",
        "rel",
        "edrel",
        "cens",
        "censor",
        "d.time",
        "survtime",
        "os",
        "rfs",
    }
    for name in list_datasets(access=Access.OPEN):
        if (requires := dataset_info(name).requires) is not None:
            if importlib.util.find_spec(requires) is None:
                continue
        ds = load_dataset(name, cache_dir=network_cache)
        overlap = {c for c in ds.feature_names if c.lower() in banned}
        assert not overlap, f"{name} keeps endpoint-like covariates: {overlap}"


# ---- ARFF reader ----


ARFF_SAMPLE = """% a comment
@RELATION DataTable
@ATTRIBUTE age\tnumeric
@ATTRIBUTE grade\t{well,poor}
@ATTRIBUTE fstat\t{0,1}

@DATA
83.0,well,0
?,poor,1
70.0,?,1
"""


def test_read_arff_types_and_missing(tmp_path):
    from tausurv.datasets._arff import read_arff

    path = tmp_path / "sample.arff"
    path.write_text(ARFF_SAMPLE)
    df = read_arff(path)

    assert df.columns == ["age", "grade", "fstat"]
    assert df.height == 3
    # Tab-separated declarations must not fold the type into the name.
    assert df["age"].dtype == pl.Float64
    # A nominal whose levels are all numeric is a numeric code, not text.
    assert df["fstat"].dtype == pl.Float64
    assert df["grade"].dtype == pl.String
    assert df["age"].null_count() == 1
    assert df["grade"].null_count() == 1
    assert df["fstat"].to_list() == [0.0, 1.0, 1.0]


def test_read_arff_rejects_a_file_with_no_attributes(tmp_path):
    from tausurv.datasets._arff import read_arff

    path = tmp_path / "empty.arff"
    path.write_text("@RELATION x\n@DATA\n1,2\n")
    with pytest.raises(ValueError, match="no @attribute"):
        read_arff(path)


def test_cause_specific_variants_exist_for_every_competing_risks_study():
    """Papers report cause-specific numbers, so the reading must be reachable.

    The base entry of a competing-risks cohort carries the any-cause
    indicator, which is rarely what a published number means -- "Melanoma"
    in a benchmark table is melanoma death, not any death. Each cause gets
    its own single-event entry so the right one can be named.
    """
    from tausurv.datasets._registry import _COMPETING_RISKS

    names = set(list_datasets())
    for study, n_causes in _COMPETING_RISKS.items():
        assert study in names, study
        for k in range(1, n_causes + 1):
            assert f"{study}:cause{k}" in names


def test_no_name_carries_two_variant_suffixes():
    """`base:variant` must stay readable back into exactly two parts."""
    for name in list_datasets():
        assert name.count(":") <= 1, name


@pytest.mark.network
def test_cause_specific_partitions_the_competing_risks_events(network_cache):
    """The causes must add up: summing them recovers the any-cause indicator."""
    from tausurv.datasets._registry import _COMPETING_RISKS

    for study, n_causes in _COMPETING_RISKS.items():
        base = load_dataset(study, cache_dir=network_cache)
        total = np.zeros(base.n, dtype=np.int64)
        for k in range(1, n_causes + 1):
            part = load_dataset(f"{study}:cause{k}", cache_dir=network_cache)
            assert part.n == base.n, study
            assert not part.is_competing_risks
            total += part.event_indicator.astype(np.int64)
        assert np.array_equal(total, base.event_indicator.astype(np.int64)), study
