"""
Truth-table access: fetch, cache, verify, and partition-load the canonical ground truth.

The truth table is ~164 MB (an R2 object / release asset), so:
  - `ensure_local_truth()` fetches it once, verifies its sha256 against the committed
    manifest, and caches it under a content-addressed name so repeated runs (PRs, VM
    cycles) reuse it instead of re-downloading 164 MB.
  - `load_partition(setting, target)` reads ONLY that (setting, target) slice via Parquet
    predicate pushdown — the full 18.4M-row table is never loaded into memory.

Source resolution for `ensure_local_truth`:
  - an explicit local path that already matches the manifest sha256 (used in tests/dev),
  - else the env var TRUTH_TABLE_URL (http(s):// or file://) downloaded + verified + cached.
"""

import hashlib
import json
import os
import shutil
import urllib.request

import pandas as pd
import pyarrow.parquet as pq

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(_REPO_ROOT, "reference", "truth_table_manifest.json")
ENV_URL = "TRUTH_TABLE_URL"
ENV_CACHE = "FLUX_TRUTH_CACHE"


def _default_cache_dir():
    return os.environ.get(ENV_CACHE) or os.path.join(
        os.path.expanduser("~"), ".cache", "fluxtrapolation"
    )


def load_manifest(manifest_path=DEFAULT_MANIFEST):
    with open(manifest_path) as f:
        return json.load(f)


def expected_sha256(manifest_path=DEFAULT_MANIFEST):
    return load_manifest(manifest_path)["sha256"]


def expected_combo(setting, target, manifest_path=DEFAULT_MANIFEST):
    """Per-(setting,target) expected stats from the manifest: {'rows','sites',...}."""
    return load_manifest(manifest_path)["combos"][f"{setting}/{target}"]


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download(source, dest, chunk=1 << 20):
    """Fetch `source` (http(s)://, file://, or local path) to `dest`."""
    if source.startswith(("http://", "https://", "file://")):
        with urllib.request.urlopen(source) as r, open(dest, "wb") as out:
            shutil.copyfileobj(r, out, length=chunk)
    else:  # plain local path
        shutil.copy2(source, dest)


def ensure_local_truth(source=None, *, manifest_path=DEFAULT_MANIFEST, cache_dir=None):
    """Return a local path to a truth table whose sha256 matches the manifest.

    Verifies integrity before returning. Caches under a content-addressed filename so the
    download happens at most once per truth-table version.
    """
    expected = expected_sha256(manifest_path)
    source = source or os.environ.get(ENV_URL)

    # 1. An explicit local file that already matches — use it directly (tests/dev/VM mount).
    if source and os.path.exists(source) and sha256_of(source) == expected:
        return source

    # 2. A cached copy from a previous run.
    cache_dir = cache_dir or _default_cache_dir()
    cached = os.path.join(cache_dir, f"truth_table_{expected[:16]}.parquet")
    if os.path.exists(cached) and sha256_of(cached) == expected:
        return cached

    # 3. Fetch from source, verify, cache atomically.
    if not source:
        raise RuntimeError(
            f"No truth table available: set {ENV_URL} (R2/release URL) or pass `source`."
        )
    os.makedirs(cache_dir, exist_ok=True)
    tmp = cached + ".tmp"
    _download(source, tmp)
    got = sha256_of(tmp)
    if got != expected:
        os.remove(tmp)
        raise RuntimeError(f"Truth table sha256 mismatch: got {got}, expected {expected}")
    os.replace(tmp, cached)
    return cached


def load_partition(setting, target, truth_path, columns=("site_id", "time", "y_true")):
    """Load only the (setting, target) slice of the truth table via predicate pushdown."""
    table = pq.read_table(
        truth_path,
        columns=list(columns),
        filters=[("setting", "=", setting), ("target", "=", target)],
    )
    df = table.to_pandas()
    if df.empty:
        raise ValueError(f"Truth table has no rows for setting={setting!r} target={target!r}")
    if "site_id" in df.columns:
        df["site_id"] = df["site_id"].astype(str)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    return df
