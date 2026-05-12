"""Cache directory resolution and integrity-verified downloads."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import platformdirs
from tqdm.auto import tqdm

from tausurv.datasets._errors import DatasetIntegrityError, OfflineModeError

_OFFLINE_ENV = "TAUSURV_OFFLINE"
_CACHE_ENV = "TAUSURV_DATA"
_CHUNK = 65536


def resolve_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Resolve the cache root.

    Priority: explicit ``cache_dir`` > ``TAUSURV_DATA`` env > platformdirs default.
    """
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    env = os.environ.get(_CACHE_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return Path(platformdirs.user_cache_dir("tausurv")) / "datasets"


def offline_mode() -> bool:
    """True if ``TAUSURV_OFFLINE`` is set to a truthy value."""
    return os.environ.get(_OFFLINE_ENV, "") not in ("", "0", "false", "False")


def cached_path(name: str, url: str, cache_dir: str | Path | None = None) -> Path:
    """Path the raw file lives at after a successful fetch."""
    return resolve_cache_dir(cache_dir) / name / Path(urlparse(url).path).name


def fetch_to_cache(
    name: str,
    url: str,
    expected_sha256: str,
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> Path:
    """Download ``url`` to the cache, verify SHA256, and return the file path.

    No-op if the file is already cached and ``force_download`` is False.
    Bytes are written to ``<target>.part`` and atomically renamed only after
    the digest matches; a mismatch removes the partial and raises
    :class:`DatasetIntegrityError`.
    """
    target = cached_path(name, url, cache_dir)
    if target.exists() and not force_download:
        return target
    if offline_mode():
        raise OfflineModeError(name, url)

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    sha = hashlib.sha256()

    with urllib.request.urlopen(url, timeout=30) as response:
        total = int(response.headers.get("Content-Length", "0") or 0) or None
        bar = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"tausurv.datasets/{name}",
            leave=False,
        )
        try:
            with open(partial, "wb") as fh:
                while chunk := response.read(_CHUNK):
                    fh.write(chunk)
                    sha.update(chunk)
                    bar.update(len(chunk))
        finally:
            bar.close()

    actual = sha.hexdigest()
    if actual != expected_sha256:
        partial.unlink(missing_ok=True)
        raise DatasetIntegrityError(name, expected_sha256, actual)

    partial.replace(target)
    _write_metadata(target, name=name, url=url, sha256=expected_sha256)
    return target


def _write_metadata(file_path: Path, *, name: str, url: str, sha256: str) -> None:
    meta = {
        "name": name,
        "url": url,
        "sha256": sha256,
        "filename": file_path.name,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (file_path.parent / "metadata.json").write_text(json.dumps(meta, indent=2))
