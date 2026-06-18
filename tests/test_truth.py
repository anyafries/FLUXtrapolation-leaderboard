"""Tests for truth-table fetch/cache/verify + partition loading."""

import json
import os

import pytest

from server import truth as truth_mod
from conftest import SETTINGS, TARGETS, SITES, N_HOURS


def test_load_partition_returns_only_that_slice(truth_fixture):
    truth_path, _, _ = truth_fixture
    df = truth_mod.load_partition("spatial-easy40", "GPP", truth_path)
    assert len(df) == len(SITES) * N_HOURS
    assert set(df.columns) == {"site_id", "time", "y_true"}
    assert str(df["time"].dtype).startswith("datetime64")


def test_ensure_local_truth_accepts_matching_local_file(truth_fixture):
    truth_path, manifest_path, _ = truth_fixture
    out = truth_mod.ensure_local_truth(truth_path, manifest_path=manifest_path)
    assert out == truth_path  # already matches -> used directly, no copy


def test_ensure_local_truth_rejects_sha_mismatch(truth_fixture, tmp_path):
    truth_path, manifest_path, _ = truth_fixture
    bad_manifest = os.path.join(tmp_path, "bad_manifest.json")
    with open(manifest_path) as f:
        m = json.load(f)
    m["sha256"] = "0" * 64  # wrong
    with open(bad_manifest, "w") as f:
        json.dump(m, f)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        truth_mod.ensure_local_truth(truth_path, manifest_path=bad_manifest,
                                     cache_dir=os.path.join(tmp_path, "cache"))


def test_ensure_local_truth_downloads_and_caches(truth_fixture, tmp_path):
    truth_path, manifest_path, _ = truth_fixture
    cache_dir = os.path.join(tmp_path, "cache")
    url = "file://" + os.path.abspath(truth_path)  # simulate a remote source
    out = truth_mod.ensure_local_truth(url, manifest_path=manifest_path, cache_dir=cache_dir)
    assert os.path.dirname(out) == cache_dir
    assert truth_mod.sha256_of(out) == truth_mod.expected_sha256(manifest_path)
    # second call reuses the cached copy (same path, no re-download needed)
    out2 = truth_mod.ensure_local_truth(url, manifest_path=manifest_path, cache_dir=cache_dir)
    assert out2 == out


def test_expected_combo_reads_manifest(truth_fixture):
    _, manifest_path, _ = truth_fixture
    combo = truth_mod.expected_combo("TA40", "NEE", manifest_path=manifest_path)
    assert combo["rows"] == len(SITES) * N_HOURS
    assert combo["sites"] == len(SITES)
