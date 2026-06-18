"""Unit tests for the standalone validator: a clean submission passes; each check rejects
its corresponding failure mode."""

import numpy as np
import pandas as pd

from server.validation import validate_submission, ValidationConfig, parse_raw_filename
from conftest import write_submission, MODEL, STRATEGY


def _validate(paths, truth_path, **kw):
    kw.setdefault("model_id", MODEL)
    kw.setdefault("val_strategy", STRATEGY)
    return validate_submission(paths, truth_path=truth_path, **kw)


def _check(report, name):
    return next(c for c in report.checks if c.name == name)


# --- happy path -----------------------------------------------------------------------

def test_clean_submission_passes(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    paths = write_submission(tmp_path, truth_df)
    report = _validate(paths, truth_path)
    assert report.passed, report.to_markdown()
    assert all(c.passed for c in report.checks)


# --- CHECK 1: files -------------------------------------------------------------------

def test_missing_file_fails(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    paths = write_submission(tmp_path, truth_df, skip={("TA40", "NEE")})
    report = _validate(paths, truth_path)
    assert not report.passed
    assert not _check(report, "files").passed
    assert "missing" in _check(report, "files").errors[0].lower()


def test_wrong_model_name_fails(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    paths = write_submission(tmp_path, truth_df, model="someoneelse")
    report = _validate(paths, truth_path, model_id=MODEL)
    assert not _check(report, "files").passed


def test_wrong_val_strategy_fails(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    paths = write_submission(tmp_path, truth_df, strategy="max")
    report = _validate(paths, truth_path, val_strategy="mean")
    assert not _check(report, "files").passed


def test_parse_filename_variants():
    assert parse_raw_filename("spatial-easy40_GPP_lr_val_mean_predictions.csv")["target"] == "GPP"
    # tolerates missing _predictions suffix
    assert parse_raw_filename("TA40_NEE_xgb_val_max.csv")["val_strategy"] == "max"
    assert parse_raw_filename("garbage.csv") is None


# --- CHECK 2: content / caps ----------------------------------------------------------

def test_nan_in_ypred_fails(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    def add_nan(df):
        df.loc[0, "y_pred"] = np.nan
        return df
    paths = write_submission(tmp_path, truth_df, edits={("time-split", "GPP"): add_nan})
    report = _validate(paths, truth_path)
    assert not _check(report, "content_caps").passed
    assert "NaN" in " ".join(_check(report, "content_caps").errors)


def test_inf_ypred_fails(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    paths = write_submission(tmp_path, truth_df,
                             edits={("TA40", "ET"): lambda d: d.assign(y_pred=np.inf)})
    report = _validate(paths, truth_path)
    assert not _check(report, "content_caps").passed


def test_magnitude_cap_fails(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    paths = write_submission(tmp_path, truth_df,
                             edits={("spatial-easy40", "NEE"): lambda d: d.assign(y_pred=1e12)})
    report = _validate(paths, truth_path, config=ValidationConfig(max_abs_ypred=1e6))
    assert not _check(report, "content_caps").passed


def test_duplicate_rows_fail(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    paths = write_submission(tmp_path, truth_df,
                             edits={("time-split", "ET"): lambda d: pd.concat([d, d.iloc[[0]]], ignore_index=True)})
    report = _validate(paths, truth_path)
    # duplicate row -> content flags it (and index sees an extra duplicate key)
    assert not _check(report, "content_caps").passed


def test_oversize_file_fails(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    paths = write_submission(tmp_path, truth_df)
    report = _validate(paths, truth_path, config=ValidationConfig(max_file_bytes=10))
    assert not _check(report, "content_caps").passed


# --- CHECK 3: index completeness (anti-cherry-pick) -----------------------------------

def test_missing_rows_fail(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    paths = write_submission(tmp_path, truth_df,
                             edits={("spatial-easy40", "GPP"): lambda d: d.iloc[:-3]})  # drop 3 rows
    report = _validate(paths, truth_path)
    assert not _check(report, "index_completeness").passed
    assert "missing" in " ".join(_check(report, "index_completeness").errors).lower()


def test_extra_rows_fail(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    def add_extra(df):
        extra = df.iloc[[0]].copy()
        extra["time"] = "1999-01-01 00:00:00"  # not in the index
        return pd.concat([df, extra], ignore_index=True)
    paths = write_submission(tmp_path, truth_df, edits={("TA40", "GPP"): add_extra})
    report = _validate(paths, truth_path)
    assert not _check(report, "index_completeness").passed
    assert "not in the canonical index" in " ".join(_check(report, "index_completeness").errors)


# --- CHECK 4: y_true integrity --------------------------------------------------------

def test_ytrue_mismatch_fails(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    paths = write_submission(tmp_path, truth_df,
                             edits={("time-split", "NEE"): lambda d: d.assign(y_true=d["y_true"] + 5.0)})
    report = _validate(paths, truth_path)
    assert not _check(report, "ytrue_integrity").passed
    assert "disagrees with lr truth" in " ".join(_check(report, "ytrue_integrity").errors)


def test_ytrue_within_tolerance_passes(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    # tiny perturbation within tolerance
    paths = write_submission(tmp_path, truth_df,
                             edits={("time-split", "NEE"): lambda d: d.assign(y_true=d["y_true"] + 1e-9)})
    report = _validate(paths, truth_path)
    assert _check(report, "ytrue_integrity").passed, report.to_markdown()


# --- CHECK 5: ownership ---------------------------------------------------------------

def test_ownership_existing_mismatch_fails(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    paths = write_submission(tmp_path, truth_df)
    report = _validate(paths, truth_path, owner_token="tok-B", recorded_owner="tok-A")
    assert not _check(report, "ownership").passed


def test_ownership_existing_match_passes(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    paths = write_submission(tmp_path, truth_df)
    report = _validate(paths, truth_path, owner_token="tok-A", recorded_owner="tok-A")
    assert report.passed, report.to_markdown()


def test_ownership_new_model_passes(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    paths = write_submission(tmp_path, truth_df)
    report = _validate(paths, truth_path, owner_token="tok-new", recorded_owner=None)
    assert _check(report, "ownership").passed


# --- report shape ---------------------------------------------------------------------

def test_report_is_machine_readable(truth_fixture, tmp_path):
    truth_path, _, truth_df = truth_fixture
    paths = write_submission(tmp_path, truth_df, skip={("TA40", "NEE")})
    d = _validate(paths, truth_path).to_dict()
    assert d["passed"] is False
    assert {c["name"] for c in d["checks"]} == {
        "files", "content_caps", "index_completeness", "ytrue_integrity", "ownership"}
