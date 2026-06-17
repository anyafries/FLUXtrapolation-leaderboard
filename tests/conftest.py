"""Shared fixtures: a tiny synthetic truth table + matching submission files.

Kept deliberately small (2 sites x 24 hours x 9 combos) so the validation suite is fast and
never touches the real 164 MB truth table.
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

SETTINGS = ["time-split", "spatial-easy40", "TA40"]
TARGETS = ["GPP", "ET", "NEE"]
SITES = ["AR-CCg", "AU-Dry"]
N_HOURS = 24
MODEL = "demo"
STRATEGY = "mean"

_TARGET_OFFSET = {"GPP": 10.0, "ET": 3.0, "NEE": -1.0}


def _truth_value(target, site, t):
    """Deterministic y_true; depends on (target, site, time) but NOT setting (as in real data)."""
    site_off = SITES.index(site) * 0.5
    return _TARGET_OFFSET[target] + site_off + np.sin(t.hour / 24.0 * 2 * np.pi)


def _build_truth_df():
    times = pd.date_range("2020-06-01", periods=N_HOURS, freq="h")
    rows = []
    for s in SETTINGS:
        for tgt in TARGETS:
            for site in SITES:
                for t in times:
                    rows.append((s, tgt, site, t, _truth_value(tgt, site, t)))
    return pd.DataFrame(rows, columns=["setting", "target", "site_id", "time", "y_true"])


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


@pytest.fixture(scope="session")
def truth_fixture(tmp_path_factory):
    """Returns (truth_path, manifest_path, truth_df) for a tiny verified truth table."""
    d = tmp_path_factory.mktemp("truth")
    truth_df = _build_truth_df()
    truth_path = os.path.join(d, "truth_table.parquet")
    truth_df.to_parquet(truth_path, index=False)

    combos = {}
    for s in SETTINGS:
        for t in TARGETS:
            sub = truth_df[(truth_df.setting == s) & (truth_df.target == t)]
            combos[f"{s}/{t}"] = {"rows": int(len(sub)), "sites": int(sub.site_id.nunique())}
    manifest = {"truth_model": "lr", "val_strategy": "mean", "combos": combos,
                "total_rows": int(len(truth_df)), "sha256": _sha256(truth_path)}
    manifest_path = os.path.join(d, "truth_table_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return truth_path, manifest_path, truth_df


def write_submission(out_dir, truth_df, model=MODEL, strategy=STRATEGY,
                     edits=None, skip=None):
    """Write the 9 raw prediction files from the truth fixture.

    edits: dict[(setting,target) -> fn(df)->df] to mutate a specific file's DataFrame.
    skip:  set of (setting,target) to omit entirely.
    Returns the list of written file paths.
    """
    edits = edits or {}
    skip = skip or set()
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for s in SETTINGS:
        for t in TARGETS:
            if (s, t) in skip:
                continue
            sub = truth_df[(truth_df.setting == s) & (truth_df.target == t)].copy()
            df = pd.DataFrame({
                "y_true": sub["y_true"].to_numpy(),
                "y_pred": sub["y_true"].to_numpy() + 0.25,  # finite, plausible
                "env": sub["site_id"].to_numpy(),
                "site_id": sub["site_id"].to_numpy(),
                "time": sub["time"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
            })
            if (s, t) in edits:
                df = edits[(s, t)](df)
            name = f"{s}_{t}_{model}_val_{strategy}_predictions.csv"
            path = os.path.join(out_dir, name)
            df.to_csv(path, index=False)
            paths.append(path)
    return paths
