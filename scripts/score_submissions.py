#!/usr/bin/env python3
"""
Score every still-`pending` submission on main — runs in the score-and-publish GitHub Action
(Phase 4, VM-free). Scoring eval.py on a 9-file submission takes minutes, well under Action limits.

For each submissions/*/metadata.yaml with status: pending:
  1. download its 9 raw files from R2 using the `r2_key`s in metadata (same R2 read creds the
     validate workflow uses),
  2. join the canonical lr truth table (ensure_local_truth + manifest-sha verify, reusing the
     cached copy), discard the submitted y_true, score y_pred against truth via eval.py,
  3. write the 9 per-scale metric CSVs beside metadata.yaml,
  4. flip status -> scored: record `sha256` per file, RETAIN `r2_key`.

Idempotent: `scored` submissions are skipped, so a missed/failed run self-heals on the next push.
A submission that fails to score stays `pending` (loud, non-zero exit) and is retried next time.

Explicitly does NOT (archiving is now a separate manual/scheduled sweep):
  - delete or move the raw R2 files (they remain in incoming/),
  - touch any archive / ArchiveBackend code.

The leaderboard rebuild + commit are done by the workflow (scripts/build_leaderboard.py); this
script only produces metrics + flips metadata.
"""

import os
import sys
import glob

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from server import metadata as meta_mod  # noqa: E402
from server.scoring import score_predictions  # noqa: E402
from server.objectstore import get_object_store  # noqa: E402
from server.truth import ensure_local_truth, sha256_of  # noqa: E402

SUBMISSIONS_DIR = os.path.join(_REPO_ROOT, "submissions")


def find_pending():
    """All submissions/*/metadata.yaml whose status is `pending`, sorted for stable ordering."""
    pending = []
    for path in sorted(glob.glob(os.path.join(SUBMISSIONS_DIR, "*", "metadata.yaml"))):
        try:
            meta = meta_mod.load_metadata(path)
        except Exception as e:
            print(f"  ! {path}: could not read metadata ({e}); skipping", file=sys.stderr)
            continue
        if meta.get("status") == meta_mod.STATUS_PENDING:
            pending.append((path, meta))
    return pending


def metric_filename(setting, target, model_id, val_strategy):
    """Bare metric-CSV name the leaderboard builder parses (no `_predictions`)."""
    return f"{setting}_{target}_{model_id}_val_{val_strategy}.csv"


def score_one(meta_path, meta, store, truth_path, workdir):
    """Score one pending submission: write 9 metric CSVs + rewrite metadata.yaml as scored.

    Raises on any failure (missing R2 object, coverage error, …) so the caller can leave the
    submission `pending` and surface it. Never deletes R2 objects.
    """
    model_id = meta["model_id"]
    val_strategy = meta["val_strategy"]
    out_dir = os.path.dirname(meta_path)

    scored_files = []
    for f in meta["files"]:
        setting, target, r2_key = f["setting"], f["target"], f["r2_key"]
        dest = os.path.join(workdir, f["filename"])
        store.get(r2_key, dest)  # download raw from R2 (raw stays in R2; we only read)

        df = pd.read_csv(dest)
        metrics = score_predictions(df, setting, target, model_id, truth_path=truth_path)
        metrics.to_csv(os.path.join(out_dir, metric_filename(setting, target, model_id, val_strategy)),
                       index=False)

        scored_files.append({
            "filename": f["filename"], "setting": setting, "target": target,
            "r2_key": r2_key,            # retained: raw persists in R2 incoming/
            "rows": int(len(df)),
            "sha256": sha256_of(dest),   # integrity of the bytes we scored
        })

    new_meta = meta_mod.build_metadata(
        model_id=model_id, val_strategy=val_strategy, owner=meta.get("owner"),
        files=scored_files, status=meta_mod.STATUS_SCORED,
        display_name=meta.get("display_name"), email=meta.get("email"),
        description=meta.get("description"), code_url=meta.get("code_url"),
        paper_url=meta.get("paper_url"), is_baseline=meta.get("is_baseline", False),
        submitted_at=meta.get("submitted_at"),
    )
    meta_mod.write_metadata(meta_path, new_meta)


def main():
    import tempfile

    pending = find_pending()
    if not pending:
        print("No pending submissions — nothing to score.")
        return 0

    truth_path = ensure_local_truth()  # fetch/cache + sha256-verify (TRUTH_TABLE_URL)
    store = get_object_store()         # R2 in the Action (OBJECTSTORE_BACKEND=r2)

    scored, failed = [], []
    for meta_path, meta in pending:
        label = f"{meta.get('model_id')}_val_{meta.get('val_strategy')}"
        workdir = tempfile.mkdtemp(prefix="score_")
        try:
            score_one(meta_path, meta, store, truth_path, workdir)
            print(f"  scored {label} ({len(meta['files'])} files)")
            scored.append(label)
        except Exception as e:
            print(f"  ! FAILED to score {label}: {e}", file=sys.stderr)
            failed.append(label)
        finally:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)

    print(f"\nScored {len(scored)}: {scored}")
    if failed:
        print(f"FAILED {len(failed)} (left pending, retried next push): {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
