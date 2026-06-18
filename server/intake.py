"""
Intake validation driver — what the validate-pr Action runs to gate auto-merge.

A submission PR adds exactly one file: submissions/{model_id}_val_{strategy}/metadata.yaml
(status: pending), whose file entries point at the raw CSVs sitting in the R2 dock. This driver:

  1. checks the metadata schema (intake stage),
  2. downloads the 9 raw files from R2 into a sandbox temp dir,
  3. runs the full validator (server.validation) against the canonical truth table,
  4. emits a machine-readable report (gates merge) + markdown (PR comment).

The core (`validate_intake`) takes an ObjectStore and is fully testable with the local backend;
`main` adds the Action glue (changed-file guard, recorded-owner lookup via git, env config).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from server import metadata as meta_mod  # noqa: E402
from server.validation import validate_submission, CheckResult, ValidationReport  # noqa: E402
from server.objectstore import get_object_store  # noqa: E402


def _report_with_metadata_failure(model_id, val_strategy, errors):
    c = CheckResult("metadata", passed=False, errors=errors)
    return ValidationReport(model_id=model_id or "?", val_strategy=val_strategy or "?", checks=[c])


def validate_intake(metadata_path, *, object_store, recorded_owner=None,
                    truth_path=None, truth_source=None, workdir=None, config=None):
    """Validate one intake metadata.yaml + its R2-hosted raw files. Returns a ValidationReport."""
    meta = meta_mod.load_metadata(metadata_path)
    model_id = meta.get("model_id")
    val_strategy = meta.get("val_strategy")

    schema_errors = meta_mod.validate_metadata(meta)
    if meta.get("status") != meta_mod.STATUS_PENDING:
        schema_errors.append(f"status must be `{meta_mod.STATUS_PENDING}` for a submission PR")
    if schema_errors:
        return _report_with_metadata_failure(model_id, val_strategy, schema_errors)

    # Download the raw files from the R2 dock into the sandbox.
    own_tmp = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="intake_")
    try:
        local_paths = []
        for f in meta["files"]:
            dest = os.path.join(workdir, f["filename"])
            try:
                object_store.get(f["r2_key"], dest)
            except Exception as e:  # missing/unreachable object -> metadata-level failure
                return _report_with_metadata_failure(
                    model_id, val_strategy, [f"could not fetch `{f['r2_key']}` from R2: {e}"])
            local_paths.append(dest)

        report = validate_submission(
            local_paths, model_id=model_id, val_strategy=val_strategy,
            owner_token=meta.get("owner"), recorded_owner=recorded_owner,
            truth_path=truth_path, truth_source=truth_source, config=config,
        )
        # Record that the metadata schema passed, as an explicit check.
        report.checks.insert(0, CheckResult("metadata", passed=True))
        return report
    finally:
        if own_tmp:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)


# --------------------------------------------------------------------------------------
# Action glue
# --------------------------------------------------------------------------------------

def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)


def find_recorded_owner(model_id, base_ref):
    """Owner hash recorded for `model_id` on the base branch (any val_strategy), or None."""
    ls = _git("ls-tree", "-r", "--name-only", base_ref, "submissions/")
    if ls.returncode != 0:
        return None
    prefix = f"submissions/{model_id}_val_"
    for path in ls.stdout.splitlines():
        if path.startswith(prefix) and path.endswith("/metadata.yaml"):
            show = _git("show", f"{base_ref}:{path}")
            if show.returncode == 0:
                import yaml
                return (yaml.safe_load(show.stdout) or {}).get("owner")
    return None


def _guard_changed_files(changed_files):
    """A submission PR may only add a single submissions/{id}_val_{s}/metadata.yaml."""
    meta_files = [f for f in changed_files if f.endswith("/metadata.yaml")
                  and f.replace("\\", "/").startswith("submissions/")]
    other = [f for f in changed_files if f not in meta_files]
    return meta_files, other


def main():
    ap = argparse.ArgumentParser(description="Validate an intake submission PR")
    ap.add_argument("changed_files", nargs="*", help="files changed in the PR")
    ap.add_argument("--base-ref", default="origin/main")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    meta_files, other = _guard_changed_files(args.changed_files)
    if other:
        rep = _report_with_metadata_failure(None, None, [
            "a submission PR may only add `submissions/{model_id}_val_{strategy}/metadata.yaml`; "
            f"these files are not allowed: {other}"])
    elif len(meta_files) != 1:
        rep = _report_with_metadata_failure(None, None, [
            f"expected exactly one metadata.yaml, got {len(meta_files)}: {meta_files}"])
    else:
        path = meta_files[0]
        meta = meta_mod.load_metadata(path)
        recorded = find_recorded_owner(meta.get("model_id"), args.base_ref)
        rep = validate_intake(path, object_store=get_object_store(), recorded_owner=recorded,
                              truth_source=os.environ.get("TRUTH_TABLE_URL"))

    print(rep.to_markdown())
    if args.json_out:
        with open(args.json_out, "w") as f:
            f.write(rep.to_json(indent=2))
    sys.exit(0 if rep.passed else 1)


if __name__ == "__main__":
    main()
