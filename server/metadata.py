"""
metadata.yaml schema for a submission, across its lifecycle.

One metadata.yaml lives in each submission folder `submissions/{model_id}_val_{strategy}/`.
It has two stages, distinguished by `status`:

  status: pending   — written by the relay (Worker) at intake. The 9 raw files live in the
                      transient R2 dock; each file entry carries its `r2_key`. No metric CSVs
                      committed yet. This is what the validate-pr Action checks before merge.
  status: scored    — finalised by the VM after scoring. Each file entry gains `sha256` +
                      `archive_pointer` (keep-forever archive), the `r2_key` is dropped (file
                      deleted from R2), and the 9 metric CSVs sit beside metadata.yaml.
                      Maintainer-uploaded baselines (e.g. lr) are written directly as `scored`.

Identity / ownership:
  - `model_id` is owned on first submission. `owner` is sha256(owner_token) — never the raw
    token (the repo is public). The Worker's KV is authoritative for issuing/checking tokens;
    storing the hash lets the validator do defense-in-depth. Baselines use owner "maintainer".
  - `display_name` / `email` are for display + contact only; not verified.

Reused by: build_baseline_lr.py, the relay/Worker (intake), validation.py, process_submission.py.
"""

import datetime as _dt
import hashlib

import yaml

# Top-level fields (all stages).
TOP_FIELDS = [
    "model_id", "display_name", "email", "description", "code_url", "paper_url",
    "owner", "val_strategy", "submitted_at", "is_baseline", "status",
]

STATUS_PENDING = "pending"
STATUS_SCORED = "scored"

# Per-file fields. Core are always present; the rest are stage-specific.
CORE_FILE_FIELDS = ["filename", "setting", "target"]
ALLOWED_FILE_FIELDS = CORE_FILE_FIELDS + ["rows", "r2_key", "sha256", "archive_pointer"]
# Required by stage (in addition to core):
STAGE_REQUIRED = {
    STATUS_PENDING: ["r2_key"],                  # raw file is in R2, awaiting scoring
    STATUS_SCORED: ["sha256", "archive_pointer"],  # raw file archived; hash recorded
}


def utcnow_iso():
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def owner_hash(owner_token):
    """Non-secret owner identifier stored in metadata: sha256 of the raw owner token."""
    return hashlib.sha256(owner_token.encode()).hexdigest()


def _file_entry(f):
    """Keep only allowed keys, preserving whatever stage-specific fields are present."""
    return {k: f[k] for k in ALLOWED_FILE_FIELDS if k in f}


def build_metadata(model_id, val_strategy, owner, files, *, status=STATUS_SCORED,
                   display_name=None, email=None, description=None,
                   code_url=None, paper_url=None, is_baseline=False, submitted_at=None):
    """Assemble a metadata dict. `files` is a list of dicts; each must carry CORE_FILE_FIELDS
    plus the fields STAGE_REQUIRED for `status`."""
    if status not in STAGE_REQUIRED:
        raise ValueError(f"unknown status {status!r}")
    required = CORE_FILE_FIELDS + STAGE_REQUIRED[status]
    for f in files:
        missing = [k for k in required if k not in f]
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
        "status": status,
        "files": [_file_entry(f) for f in files],
    }


def build_intake_metadata(model_id, val_strategy, owner_token, files, **kw):
    """Convenience for the relay: status=pending, owner stored as a hash, files carry r2_key."""
    return build_metadata(model_id, val_strategy, owner_hash(owner_token), files,
                          status=STATUS_PENDING, **kw)


def write_metadata(path, meta):
    with open(path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False, default_flow_style=False)


def load_metadata(path):
    with open(path) as f:
        return yaml.safe_load(f)


def validate_metadata(meta):
    """Return a list of human-readable problems (empty == valid). Stage-aware."""
    errors = []
    for k in TOP_FIELDS:
        if k not in meta:
            errors.append(f"missing top-level field: {k}")
    status = meta.get("status")
    if status not in STAGE_REQUIRED:
        errors.append(f"invalid status: {status!r}")
    files = meta.get("files")
    if not isinstance(files, list) or not files:
        errors.append("`files` must be a non-empty list")
        return errors
    required = CORE_FILE_FIELDS + STAGE_REQUIRED.get(status, [])
    for f in files:
        for k in required:
            if k not in f:
                errors.append(f"file {f.get('filename')!r} missing field: {k}")
    return errors
