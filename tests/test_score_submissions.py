"""Tests for scripts/score_submissions.py — the VM-free scoring driver.

Uses the tiny synthetic truth table + raw files from conftest, a LocalObjectStore as the R2
stand-in, and a fake submissions/ tree, so nothing touches the network or the 164 MB truth table.
"""

import importlib.util
import os
import sys

import pandas as pd
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from server import metadata as meta_mod
from server.objectstore import LocalObjectStore

from conftest import write_submission, SETTINGS, TARGETS, STRATEGY

# Load the script as a module (it lives in scripts/, not an importable package).
_SPEC = importlib.util.spec_from_file_location(
    "score_submissions", os.path.join(_REPO_ROOT, "scripts", "score_submissions.py"))
score_submissions = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(score_submissions)

MODEL = "demo"


def _seed(tmp_path, truth_df, store_root, subs_dir, model=MODEL):
    """Put the 9 raw files in the (local) R2 store and write a pending metadata.yaml."""
    raw_dir = tmp_path / "raw"
    paths = write_submission(str(raw_dir), truth_df, model=model)
    store = LocalObjectStore(root=str(store_root))
    files = []
    for p in paths:
        fn = os.path.basename(p)
        setting, target = fn.split("_")[0], fn.split("_")[1]
        r2_key = f"incoming/{model}_val_{STRATEGY}/{fn}"
        store.put(r2_key, p)
        files.append({"filename": fn, "setting": setting, "target": target, "r2_key": r2_key})

    out_dir = subs_dir / f"{model}_val_{STRATEGY}"
    out_dir.mkdir(parents=True)
    meta = meta_mod.build_intake_metadata(model, STRATEGY, "tok", files, display_name="Demo")
    meta_mod.write_metadata(str(out_dir / "metadata.yaml"), meta)
    return store, out_dir


def test_scores_pending_then_idempotent(tmp_path, truth_fixture, monkeypatch):
    truth_path, _manifest, truth_df = truth_fixture
    store_root = tmp_path / "r2"
    subs_dir = tmp_path / "submissions"
    store, out_dir = _seed(tmp_path, truth_df, store_root, subs_dir)

    # Point the script at our fake submissions/ tree + local object store + tiny truth table.
    monkeypatch.setattr(score_submissions, "SUBMISSIONS_DIR", str(subs_dir))
    monkeypatch.setattr(score_submissions, "get_object_store", lambda: store)
    monkeypatch.setattr(score_submissions, "ensure_local_truth", lambda: truth_path)

    rc = score_submissions.main()
    assert rc == 0

    # 9 metric CSVs written beside metadata.yaml, with the leaderboard's schema.
    metric_csvs = [f for f in os.listdir(out_dir) if f.endswith(".csv")]
    assert len(metric_csvs) == 9
    sample = pd.read_csv(out_dir / metric_csvs[0])
    for col in ["target", "setting", "model", "scale", "env", "rmse"]:
        assert col in sample.columns
    assert (sample["model"] == MODEL).all()

    # Metadata flipped to scored, sha256 recorded, r2_key RETAINED, no archive_pointer required.
    meta = meta_mod.load_metadata(str(out_dir / "metadata.yaml"))
    assert meta["status"] == meta_mod.STATUS_SCORED
    assert meta_mod.validate_metadata(meta) == []
    for f in meta["files"]:
        assert f["sha256"] and len(f["sha256"]) == 64
        assert f["r2_key"].startswith(f"incoming/{MODEL}_val_{STRATEGY}/")
        assert "archive_pointer" not in f

    # Raw R2 objects are NOT deleted by scoring (archiving is a separate sweep).
    for f in meta["files"]:
        assert store.exists(f["r2_key"]), f"{f['r2_key']} must remain in R2"

    # Idempotent: a second run finds nothing pending and writes no new metric CSVs.
    mtimes = {f: os.path.getmtime(out_dir / f) for f in os.listdir(out_dir)}
    assert score_submissions.main() == 0
    assert {f: os.path.getmtime(out_dir / f) for f in os.listdir(out_dir)} == mtimes


def test_failed_submission_left_pending(tmp_path, truth_fixture, monkeypatch):
    """A submission whose raw rows fall outside the truth index fails to score, stays pending,
    and the run exits non-zero — without deleting its R2 objects."""
    truth_path, _manifest, truth_df = truth_fixture
    store_root = tmp_path / "r2"
    subs_dir = tmp_path / "submissions"

    # Corrupt one file's (site_id,time) so the truth join leaves uncovered rows -> CoverageError.
    raw_dir = tmp_path / "raw"
    bad = {(SETTINGS[0], TARGETS[0]): lambda df: df.assign(site_id="ZZ-NOPE")}
    paths = write_submission(str(raw_dir), truth_df, model=MODEL, edits=bad)
    store = LocalObjectStore(root=str(store_root))
    files = []
    for p in paths:
        fn = os.path.basename(p)
        s, t = fn.split("_")[0], fn.split("_")[1]
        key = f"incoming/{MODEL}_val_{STRATEGY}/{fn}"
        store.put(key, p)
        files.append({"filename": fn, "setting": s, "target": t, "r2_key": key})
    out_dir = subs_dir / f"{MODEL}_val_{STRATEGY}"
    out_dir.mkdir(parents=True)
    meta_mod.write_metadata(str(out_dir / "metadata.yaml"),
                            meta_mod.build_intake_metadata(MODEL, STRATEGY, "tok", files))

    monkeypatch.setattr(score_submissions, "SUBMISSIONS_DIR", str(subs_dir))
    monkeypatch.setattr(score_submissions, "get_object_store", lambda: store)
    monkeypatch.setattr(score_submissions, "ensure_local_truth", lambda: truth_path)

    assert score_submissions.main() == 1  # loud failure
    meta = meta_mod.load_metadata(str(out_dir / "metadata.yaml"))
    assert meta["status"] == meta_mod.STATUS_PENDING  # left for retry
    assert store.exists(files[0]["r2_key"])  # R2 objects untouched
