"""Adversarial integration test against the REAL lr submission + truth table.

The hermetic versions of these five failure modes are asserted with tiny synthetic fixtures
in test_validation.py (fast, CI-safe). This module re-runs them on the actual 1.3 GB lr files
and the 164 MB truth table, asserting each tamper is caught by *exactly* its own check.

Slow + data-dependent, so it is OPT-IN: it only runs when FLUX_REAL_TESTS=1 and the gitignored
raw files + truth table are present. Otherwise it skips (CI and fresh clones stay green).

    FLUX_REAL_TESTS=1 pytest tests/test_adversarial_real.py -s
"""

import glob
import os
import shutil
import sys

import pandas as pd
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from server.validation import validate_submission, ValidationConfig

SRC = os.path.join(_REPO, "submissions_raw", "lr")
TRUTH = os.path.join(_REPO, "reference", "truth_table.parquet")

pytestmark = pytest.mark.skipif(
    not (os.environ.get("FLUX_REAL_TESTS")
         and os.path.isdir(SRC) and glob.glob(f"{SRC}/*.csv") and os.path.exists(TRUTH)),
    reason="opt-in: set FLUX_REAL_TESTS=1 with real lr files + truth table present",
)


def _files(d):
    return sorted(glob.glob(f"{d}/*.csv"))


def _run(d, owner=None):
    rep = validate_submission(
        _files(d), model_id="lr", val_strategy="mean",
        owner_token="me", recorded_owner=owner,
        truth_path=TRUTH, config=ValidationConfig(),
    )
    failed = [c.name for c in rep.checks if not c.passed]
    print(f"\npassed={rep.passed} failed_checks={failed}\n{rep.to_markdown()}")
    return rep, failed


@pytest.fixture
def workdir(tmp_path):
    d = os.path.join(tmp_path, "tamper")
    shutil.copytree(SRC, d)
    return d


def test_clean_passes(workdir):
    rep, failed = _run(workdir)
    assert rep.passed and failed == []


def test_dropped_rows_caught_by_index(workdir):
    f = _files(workdir)[0]
    pd.read_csv(f).iloc[:-100].to_csv(f, index=False)
    _, failed = _run(workdir)
    assert failed == ["index_completeness"]


def test_tampered_ytrue_caught_by_integrity(workdir):
    f = _files(workdir)[0]
    df = pd.read_csv(f); df.loc[0, "y_true"] += 1.0; df.to_csv(f, index=False)
    _, failed = _run(workdir)
    assert failed == ["ytrue_integrity"]


def test_nan_ypred_caught_by_content(workdir):
    f = _files(workdir)[0]
    df = pd.read_csv(f); df.loc[0, "y_pred"] = float("nan"); df.to_csv(f, index=False)
    _, failed = _run(workdir)
    assert failed == ["content_caps"]


def test_missing_file_caught_by_files(workdir):
    os.unlink(_files(workdir)[0])
    _, failed = _run(workdir)
    assert failed == ["files"]


def test_owner_mismatch_caught_by_ownership(workdir):
    _, failed = _run(workdir, owner="someone_else")
    assert failed == ["ownership"]
