"""Tests for the intake driver: metadata.yaml + R2-hosted raw files -> validation report.

Uses the local object store (no R2) and the tiny truth fixture from conftest."""

import os

import pytest

from server import metadata as meta_mod
from server.intake import validate_intake
from server.objectstore import LocalObjectStore, incoming_key
from server.validation import parse_raw_filename
from conftest import write_submission, MODEL, STRATEGY


def _seed_intake(tmp_path, truth_df, model=MODEL, strategy=STRATEGY,
                 owner_token="tok", edits=None, skip=None, drop_object=None):
    """Write raw files, seed a local R2 store, and build an intake metadata.yaml.

    Returns (metadata_path, store). `drop_object`=(setting,target) omits that R2 object.
    """
    raw_dir = os.path.join(tmp_path, "raw")
    paths = write_submission(raw_dir, truth_df, model=model, strategy=strategy,
                             edits=edits, skip=skip)
    store = LocalObjectStore(root=os.path.join(tmp_path, "r2"))
    files_meta = []
    for p in paths:
        fn = os.path.basename(p)
        info = parse_raw_filename(fn)
        key = incoming_key(model, strategy, fn)
        if drop_object != (info["setting"], info["target"]):
            store.put(key, p)
        rows = sum(1 for _ in open(p)) - 1
        files_meta.append({"filename": fn, "setting": info["setting"],
                           "target": info["target"], "r2_key": key, "rows": rows})
    md = meta_mod.build_intake_metadata(model, strategy, owner_token, files_meta)
    md_path = os.path.join(tmp_path, "metadata.yaml")
    meta_mod.write_metadata(md_path, md)
    return md_path, store


def _names(report):
    return [c.name for c in report.checks if not c.passed]


def test_intake_clean_passes(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    md, store = _seed_intake(tmp_path, truth_df)
    rep = validate_intake(md, object_store=store, truth_path=truth_path, recorded_owner=None)
    assert rep.passed, rep.to_markdown()
    assert rep.checks[0].name == "metadata" and rep.checks[0].passed


def test_intake_dropped_rows_fails_index(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    md, store = _seed_intake(tmp_path, truth_df,
                             edits={("spatial-easy40", "GPP"): lambda d: d.iloc[:-2]})
    rep = validate_intake(md, object_store=store, truth_path=truth_path, recorded_owner=None)
    assert _names(rep) == ["index_completeness"]


def test_intake_missing_r2_object_fails_metadata(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    md, store = _seed_intake(tmp_path, truth_df, drop_object=("TA40", "NEE"))
    rep = validate_intake(md, object_store=store, truth_path=truth_path, recorded_owner=None)
    assert not rep.passed
    assert "could not fetch" in " ".join(rep.checks[0].errors)


def test_intake_status_must_be_pending(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    md, store = _seed_intake(tmp_path, truth_df)
    # rewrite metadata as a 'scored' doc (wrong stage for a submission PR)
    m = meta_mod.load_metadata(md)
    for f in m["files"]:
        f.pop("r2_key", None)
        f["sha256"] = "0" * 64
        f["archive_pointer"] = "file:///x"
    m["status"] = meta_mod.STATUS_SCORED
    meta_mod.write_metadata(md, m)
    rep = validate_intake(md, object_store=store, truth_path=truth_path, recorded_owner=None)
    assert not rep.passed
    assert any("status must be" in e for e in rep.checks[0].errors)


def test_intake_owner_mismatch_fails(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    md, store = _seed_intake(tmp_path, truth_df, owner_token="mine")
    # recorded owner for this model_id is a different hash -> update must be rejected
    rep = validate_intake(md, object_store=store, truth_path=truth_path,
                          recorded_owner=meta_mod.owner_hash("someone_else"))
    assert _names(rep) == ["ownership"]


def test_intake_owner_match_passes(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    md, store = _seed_intake(tmp_path, truth_df, owner_token="mine")
    rep = validate_intake(md, object_store=store, truth_path=truth_path,
                          recorded_owner=meta_mod.owner_hash("mine"))
    assert rep.passed, rep.to_markdown()
