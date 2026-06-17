"""
metadata.yaml schema for a submission.

One metadata.yaml lives in each committed submission folder
`submissions/{model_id}_val_{strategy}/`, alongside the 9 recomputed metric CSVs. It records
identity/ownership/display fields plus, per raw prediction file, the content hash and the
opaque archive pointer (where the raw file lives in the keep-forever archive).

Identity / ownership (per the plan):
  - `model_id` is owned on first submission; the relay records `owner` (an opaque token).
    Updates to the same model_id must present the same owner. Baselines use owner "maintainer".
  - `display_name` / `email` are for display + contact only; not verified.

Reused by:
  - scripts/build_baseline_lr.py   (write baseline metadata)
  - scripts/validate_submission.py (Phase 2: ownership + schema checks)
  - server/process_submission.py   (Phase 4: archive pointer + hash)
"""

import datetime as _dt

import yaml

# Top-level fields. `owner` is an opaque token; "maintainer" for maintainer-uploaded baselines.
TOP_FIELDS = [
    "model_id",
    "display_name",
    "email",
    "description",
    "code_url",
    "paper_url",
    "owner",
    "val_strategy",
    "submitted_at",
    "is_baseline",
]

# Per-file fields. `filename` is the RAW prediction file; `sha256` + `archive_pointer` describe
# where that raw file is archived. The metric CSV derived from it sits beside metadata.yaml.
FILE_FIELDS = ["filename", "setting", "target", "rows", "sha256", "archive_pointer"]


def utcnow_iso():
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def build_metadata(model_id, val_strategy, owner, files,
                   display_name=None, email=None, description=None,
                   code_url=None, paper_url=None, is_baseline=False,
                   submitted_at=None):
    """Assemble a metadata dict. `files` is a list of dicts with FILE_FIELDS keys."""
    for f in files:
        missing = [k for k in FILE_FIELDS if k not in f]
        if missing:
            raise ValueError(f"file entry {f.get('filename')!r} missing fields {missing}")
    return {
        "model_id": model_id,
        "display_name": display_name or model_id,
        "email": email,
        "description": description,
        "code_url": code_url,
        "paper_url": paper_url,
        "owner": owner,
        "val_strategy": val_strategy,
        "submitted_at": submitted_at or utcnow_iso(),
        "is_baseline": bool(is_baseline),
        "files": [{k: f[k] for k in FILE_FIELDS} for f in files],
    }


def write_metadata(path, meta):
    with open(path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False, default_flow_style=False)


def load_metadata(path):
    with open(path) as f:
        return yaml.safe_load(f)


def validate_metadata(meta):
    """Return a list of human-readable problems (empty == valid). Used by Phase 2."""
    errors = []
    for k in TOP_FIELDS:
        if k not in meta:
            errors.append(f"missing top-level field: {k}")
    files = meta.get("files")
    if not isinstance(files, list) or not files:
        errors.append("`files` must be a non-empty list")
        return errors
    for f in files:
        for k in FILE_FIELDS:
            if k not in f:
                errors.append(f"file {f.get('filename')!r} missing field: {k}")
    return errors
