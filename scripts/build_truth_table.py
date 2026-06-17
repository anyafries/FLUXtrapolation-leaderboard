"""
Build the canonical ground-truth table from the trusted `lr` baseline's raw predictions.

The `lr` baseline is evaluated on the *entire* held-out set for every (setting, target),
so its rows define both the canonical `y_true` values AND the canonical required index
(the set of (site_id, time) keys a complete submission must cover).

This script reads the gitignored raw `lr` prediction CSVs once and writes a single
Parquet table keyed by (setting, target, site_id, time) -> y_true, plus a small JSON
manifest (per-combo row/site counts, time span, and a content hash for cache verification).

IMPORTANT
  - The raw `lr` CSVs (submissions_raw/lr/) are read ONLY here. Nothing at submission
    runtime (the Action or the VM) should ever read them — only this derived table.
  - `y_true` is verified to be identical across settings for shared (site_id, time), so
    the `setting` dimension carries no extra truth information; it is retained purely so a
    single artifact can also serve as the per-(setting, target) completeness index.

Usage:
    python scripts/build_truth_table.py
    python scripts/build_truth_table.py --raw-dir submissions_raw/lr --out reference/truth_table.parquet

Exit codes:
    0 — table written and all completeness assertions passed
    1 — a required raw file was missing or an integrity assertion failed
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Canonical benchmark dimensions (must match build_leaderboard.py / validate_submission.py).
SETTINGS = ["time-split", "spatial-easy40", "TA40"]
TARGETS = ["GPP", "ET", "NEE"]

# The truth source. lr is the maintainer-uploaded trusted baseline.
TRUTH_MODEL = "lr"
TRUTH_VAL_STRATEGY = "mean"

# Columns we keep in the truth table. y_true stays float64 so recomputed metrics exactly
# reproduce the values produced from the raw lr predictions.
TRUTH_SCHEMA = pa.schema([
    ("setting", pa.string()),
    ("target", pa.string()),
    ("site_id", pa.string()),
    ("time", pa.timestamp("ns")),
    ("y_true", pa.float64()),
])

# Columns we need from each raw prediction CSV (y_pred / env are not part of the truth).
RAW_USECOLS = ["site_id", "time", "y_true"]

SIZE_COMMIT_THRESHOLD_MB = 50  # commit to git if under this; otherwise R2 / release asset.


def raw_filename(setting, target):
    return f"{setting}_{target}_{TRUTH_MODEL}_val_{TRUTH_VAL_STRATEGY}_predictions.csv"


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_one(raw_dir, setting, target):
    """Load + validate one raw lr prediction file; return a typed truth DataFrame."""
    path = os.path.join(raw_dir, raw_filename(setting, target))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required truth source missing: {path}")

    df = pd.read_csv(path, usecols=RAW_USECOLS)

    missing = [c for c in RAW_USECOLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    df["time"] = pd.to_datetime(df["time"])
    df["y_true"] = df["y_true"].astype("float64")
    df["site_id"] = df["site_id"].astype("string")

    # --- integrity assertions -------------------------------------------------
    n_dup = df.duplicated(["site_id", "time"]).sum()
    if n_dup:
        raise AssertionError(f"{path}: {n_dup} duplicate (site_id, time) keys — join key not unique")

    n_bad = (~np.isfinite(df["y_true"].to_numpy())).sum()
    if n_bad:
        raise AssertionError(f"{path}: {n_bad} non-finite y_true values — truth must be complete")

    df.insert(0, "target", target)
    df.insert(0, "setting", setting)
    return df[["setting", "target", "site_id", "time", "y_true"]]


def build(raw_dir, out_path):
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    manifest = {
        "truth_model": TRUTH_MODEL,
        "val_strategy": TRUTH_VAL_STRATEGY,
        "settings": SETTINGS,
        "targets": TARGETS,
        "combos": {},
        "total_rows": 0,
    }

    # zstd + BYTE_STREAM_SPLIT on the float column: real-valued y_true compresses far better
    # when its bytes are split into planes. Strings are dictionary-encoded automatically.
    writer = pq.ParquetWriter(
        out_path, TRUTH_SCHEMA,
        compression="zstd", compression_level=9,
        use_byte_stream_split=["y_true"],
    )
    try:
        for setting in SETTINGS:
            for target in TARGETS:
                df = load_one(raw_dir, setting, target)
                table = pa.Table.from_pandas(df, schema=TRUTH_SCHEMA, preserve_index=False)
                writer.write_table(table)

                key = f"{setting}/{target}"
                manifest["combos"][key] = {
                    "rows": int(len(df)),
                    "sites": int(df["site_id"].nunique()),
                    "time_min": str(df["time"].min()),
                    "time_max": str(df["time"].max()),
                }
                manifest["total_rows"] += int(len(df))
                print(f"  [{key}] rows={len(df):,} sites={df['site_id'].nunique()}")
    finally:
        writer.close()

    # --- completeness: every (setting, target) must be present -------------------
    expected = {f"{s}/{t}" for s in SETTINGS for t in TARGETS}
    got = set(manifest["combos"])
    if got != expected:
        raise AssertionError(f"Incomplete coverage. Missing: {sorted(expected - got)}")

    size_mb = os.path.getsize(out_path) / 1e6
    manifest["bytes"] = os.path.getsize(out_path)
    manifest["sha256"] = sha256_of(out_path)

    manifest_path = os.path.splitext(out_path)[0] + "_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {out_path} ({size_mb:.1f} MB, {manifest['total_rows']:,} rows)")
    print(f"Wrote {manifest_path}")
    print(f"sha256: {manifest['sha256']}")
    if size_mb < SIZE_COMMIT_THRESHOLD_MB:
        print(f"OK to commit to git (< {SIZE_COMMIT_THRESHOLD_MB} MB).")
    else:
        print(f"EXCEEDS {SIZE_COMMIT_THRESHOLD_MB} MB — store as R2 object / release asset and "
              f"fetch-and-cache in the Action and on the VM (verify against sha256 above).")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build the canonical ground-truth table from lr raw predictions")
    parser.add_argument("--raw-dir", default=os.path.join("submissions_raw", TRUTH_MODEL),
                        help="Directory holding the raw lr *_predictions.csv files")
    parser.add_argument("--out", default=os.path.join("reference", "truth_table.parquet"),
                        help="Output Parquet path")
    args = parser.parse_args()

    try:
        build(args.raw_dir, args.out)
    except (FileNotFoundError, AssertionError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
