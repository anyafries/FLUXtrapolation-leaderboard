"""
Register the trusted `lr` baseline as a normal submission.

For each (setting, target):
  1. Score the raw lr predictions against the truth table (identity for lr, but routed
     through the same adapter every submission uses).
  2. Write the recomputed metric CSV to submissions/lr_val_mean/.
  3. Record the raw file's content hash + canonical archive pointer in metadata.yaml.

Then verify the recomputed metrics reproduce the preserved MVP metrics (submissions_metrics/lr).

The raw lr files are NOT transferred to the archive here by default (the archive base is a
deploy-time placeholder); we record the canonical pointer + real hash. Pass --archive with a
real ARCHIVE_BASE to also perform the transfer.

Usage:
    python scripts/build_baseline_lr.py
    python scripts/build_baseline_lr.py --archive          # also move raw files into the archive
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from server.scoring import score_predictions
from server.metadata import build_metadata, write_metadata
from server.archive import get_archive_backend, submission_key, sha256_of, DEFAULT_FS_BASE

SETTINGS = ["time-split", "spatial-easy40", "TA40"]
TARGETS = ["GPP", "ET", "NEE"]
MODEL = "lr"
VAL_STRATEGY = "mean"

RAW_DIR = os.path.join(_REPO_ROOT, "submissions_raw", MODEL)
OUT_DIR = os.path.join(_REPO_ROOT, "submissions", f"{MODEL}_val_{VAL_STRATEGY}")
REF_DIR = os.path.join(_REPO_ROOT, "submissions_metrics", MODEL)

# Metrics the leaderboard actually uses; we assert these reproduce. The ratio metrics
# (relative_mae/relative_bias) are excluded — they are unstable where mean(obs) ~ 0.
VERIFY_METRICS = ["mse", "rmse", "mae", "nse", "r2_score", "bias"]
# Equality via np.isclose semantics: |a-b| <= ATOL + RTOL*|b|. ATOL absorbs differences
# between two floating-point representations of zero (e.g. IAV bias of ~1e-15 vs 0).
RTOL = 1e-6
ATOL = 1e-9


def raw_name(setting, target):
    return f"{setting}_{target}_{MODEL}_val_{VAL_STRATEGY}_predictions.csv"


def metric_name(setting, target):
    return f"{setting}_{target}_{MODEL}_val_{VAL_STRATEGY}.csv"


def verify_against_reference(got, setting, target):
    """Compare recomputed metrics to the preserved metric CSV.

    Returns (n_mismatch, worst_rel) where n_mismatch counts cells failing np.isclose
    (abs+rel tolerance, so zero-vs-zero passes) and worst_rel is the worst *significant*
    relative diff (over reference values that aren't effectively zero). None if no reference.
    """
    ref_path = os.path.join(REF_DIR, metric_name(setting, target))
    if not os.path.exists(ref_path):
        return None
    ref = pd.read_csv(ref_path)
    m = got.merge(ref, on=["scale", "env"], suffixes=("_new", "_ref"))
    n_mismatch = 0
    worst_rel = 0.0
    for c in VERIFY_METRICS:
        a = m[f"{c}_new"].to_numpy(float)
        b = m[f"{c}_ref"].to_numpy(float)
        mask = np.isfinite(a) & np.isfinite(b)
        # NaN pattern must agree too (e.g. undefined NSE on constant series).
        n_mismatch += int(np.sum(np.isfinite(a) != np.isfinite(b)))
        if mask.any():
            n_mismatch += int(np.sum(~np.isclose(a[mask], b[mask], rtol=RTOL, atol=ATOL)))
            sig = mask & (np.abs(b) > 1e-8)  # ignore effectively-zero reference values
            if sig.any():
                worst_rel = max(worst_rel, float(np.max(np.abs(a[sig] - b[sig]) / np.abs(b[sig]))))
    return n_mismatch, worst_rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", action="store_true",
                    help="Also transfer raw files into the archive (needs a real ARCHIVE_BASE)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    backend = get_archive_backend() if args.archive else None
    fs_base = os.environ.get("ARCHIVE_BASE", DEFAULT_FS_BASE)

    files_meta = []
    worst_overall = 0.0
    mismatches_overall = 0
    for setting in SETTINGS:
        for target in TARGETS:
            raw_path = os.path.join(RAW_DIR, raw_name(setting, target))
            if not os.path.exists(raw_path):
                print(f"ERROR: missing raw file {raw_path}", file=sys.stderr)
                sys.exit(1)

            # Metric CSV holds the canonical 14-column schema; val_strategy is carried by the
            # folder/filename and re-derived by build_leaderboard, so it is not a column here.
            metrics = score_predictions(raw_path, setting, target, MODEL)
            out_path = os.path.join(OUT_DIR, metric_name(setting, target))
            metrics.to_csv(out_path, index=False)

            digest = sha256_of(raw_path)
            key = submission_key(MODEL, VAL_STRATEGY, raw_name(setting, target))
            if args.archive:
                pointer = backend.store(raw_path, key)
            else:
                # Canonical deploy-target pointer (filesystem backend), not yet transferred.
                pointer = f"file://{os.path.join(fs_base, key)}"

            files_meta.append({
                "filename": raw_name(setting, target),
                "setting": setting,
                "target": target,
                "rows": int(pd.read_csv(raw_path, usecols=["time"]).shape[0]),
                "sha256": digest,
                "archive_pointer": pointer,
            })

            res = verify_against_reference(metrics, setting, target)
            if res is None:
                tag = "no ref"
            else:
                n_mismatch, worst = res
                worst_overall = max(worst_overall, worst)
                mismatches_overall += n_mismatch
                tag = f"{n_mismatch} mismatch, worst sig. rel diff {worst:.2e}"
            print(f"  {setting}/{target}: wrote {metric_name(setting, target)} ({len(metrics)} rows) — {tag}")

    meta = build_metadata(
        model_id=MODEL,
        val_strategy=VAL_STRATEGY,
        owner="maintainer",
        files=files_meta,
        display_name="Linear Regression (baseline)",
        email=None,
        description="Trusted linear-regression baseline. Canonical ground-truth + index source "
                    "for scoring (see reference/truth_table.parquet).",
        is_baseline=True,
    )
    write_metadata(os.path.join(OUT_DIR, "metadata.yaml"), meta)

    print(f"\nWrote {OUT_DIR}/ (9 metric CSVs + metadata.yaml)")
    print(f"Archive transfer: {'DONE' if args.archive else 'recorded pointer only (deploy-time)'}")
    print(f"Reproduction vs preserved metrics (RMSE/MSE/MAE/NSE/R2/bias): "
          f"{mismatches_overall} mismatched cells, worst significant rel diff {worst_overall:.2e}")
    if mismatches_overall:
        print("WARNING: some cells differ beyond tolerance — investigate before relying on reproduction.")


if __name__ == "__main__":
    main()
