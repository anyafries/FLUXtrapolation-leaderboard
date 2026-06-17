"""
Score a submission's raw predictions against the canonical lr truth table.

This is the trust boundary: we discard the submitter's `y_true` entirely and join lr's
canonical `y_true` onto their `y_pred` by (site_id, time), then run the vendored
`compute_metrics`. The submitted `y_true` is only ever used (elsewhere, in validation) for
an integrity check — never for scoring.

Reused by:
  - scripts/build_baseline_lr.py     (Phase 1, register the lr baseline)
  - server/process_submission.py     (Phase 4, the VM scoring step)
"""

import os
import sys

import pandas as pd

# Make the vendored eval pipeline importable (repo root holds utils/, eval.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.eval_utils import compute_metrics  # noqa: E402
from server.truth import load_partition  # noqa: E402

DEFAULT_TRUTH_PATH = os.path.join(_REPO_ROOT, "reference", "truth_table.parquet")

RAW_REQUIRED_COLUMNS = ["y_true", "y_pred", "env", "site_id", "time"]


class CoverageError(ValueError):
    """Raised when submission rows are not covered by the truth index."""


def load_truth(setting, target, truth_path=DEFAULT_TRUTH_PATH):
    """Load the (site_id, time) -> y_true map for one (setting, target)."""
    return load_partition(setting, target, truth_path)


def join_truth(pred_df, truth_df):
    """Replace submitted y_true with the canonical truth, joined on (site_id, time).

    Returns (joined_df, n_unmatched). `n_unmatched` counts submission rows with no truth
    row — i.e. rows outside the canonical index (extra / cherry-picked / misaligned).
    """
    pred = pred_df.copy()
    pred["site_id"] = pred["site_id"].astype(str)
    pred["time"] = pd.to_datetime(pred["time"])
    pred = pred.drop(columns=["y_true"])  # never trust the submitted truth for scoring

    merged = pred.merge(truth_df, on=["site_id", "time"], how="left", validate="many_to_one")
    n_unmatched = int(merged["y_true"].isna().sum())
    return merged, n_unmatched


def score_predictions(raw, setting, target, model_name, truth_path=DEFAULT_TRUTH_PATH,
                      require_full_coverage=True):
    """Compute the per-(scale, env) metrics for one raw prediction file/DataFrame.

    Args:
        raw: path to a raw prediction CSV, or a DataFrame with RAW_REQUIRED_COLUMNS.
        require_full_coverage: if True, every submission row must match a truth row
            (rows outside the canonical index are an error, not silently dropped).

    Returns the metrics DataFrame produced by the vendored compute_metrics.
    """
    pred_df = pd.read_csv(raw) if isinstance(raw, str) else raw
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in pred_df.columns]
    if missing:
        raise ValueError(f"raw predictions missing columns {missing}")

    truth_df = load_truth(setting, target, truth_path)
    merged, n_unmatched = join_truth(pred_df, truth_df)

    if require_full_coverage and n_unmatched:
        raise CoverageError(
            f"{n_unmatched} of {len(merged)} submission rows are outside the canonical "
            f"({setting}, {target}) index — cannot be scored against truth."
        )

    return compute_metrics(merged, model_name, setting, target)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Score one raw prediction file against the truth table")
    ap.add_argument("raw_csv")
    ap.add_argument("setting")
    ap.add_argument("target")
    ap.add_argument("model_name")
    ap.add_argument("--truth", default=DEFAULT_TRUTH_PATH)
    args = ap.parse_args()

    metrics = score_predictions(args.raw_csv, args.setting, args.target, args.model_name, args.truth)
    print(metrics.to_string())
