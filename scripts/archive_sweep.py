#!/usr/bin/env python3
"""
Manual archive sweep — move scored submissions' raw files out of the R2 loading dock.

Run this BY HAND on the VM whenever you get a "submission scored successfully" email. It is the
Phase-4 archive step that scripts/score_submissions.py deliberately leaves undone: scoring keeps
the 9 raw CSVs in R2 `incoming/` (and keeps each file's `r2_key`); this sweep copies them into a
local keep-forever archive, verifies each copy, records an `archive_pointer` per file in
metadata.yaml, and only then deletes the file from R2.

For every submission with `status: scored` (pending / in-flight submissions are NEVER touched):

  for each file (by its `r2_key` in metadata.yaml):
    1. download it from R2 into the local archive (server.archive, default backend=filesystem)
    2. VERIFY before anything is deleted:
         - downloaded size == R2's reported size            (always)
         - md5(download) == R2 ETag                         (only when the ETag is a plain MD5;
                                                              multipart ETags carry a -N suffix and
                                                              are skipped, noted in the log)
         - sha256(download) == the `sha256` scoring recorded in metadata.yaml   (always)
    3. store it in the archive and re-hash the stored copy as a final guard
    4. record `archive_pointer` (the opaque file:// URI) in metadata.yaml
    5. delete the file from R2 — per file, only after its own verification passed.

Modes (R2 DELETION is gated behind --confirm):
    (no flags)          DRY RUN: list what it would download / verify / archive / delete. No writes,
                        no downloads, no deletes. Just queries R2 sizes and prints the plan.
    --download          Do the archiving (download, verify, store, record archive_pointer) but do
                        NOT delete anything from R2. Safe way to pre-stage + verify local copies.
    --confirm           Full sweep: archive AND delete each verified file from R2.

    python scripts/archive_sweep.py                                   # preview everything scored
    python scripts/archive_sweep.py --model-id coral --val-strategy mean   # preview just one
    python scripts/archive_sweep.py --confirm                         # archive + free R2

Idempotent / re-runnable: a file whose `archive_pointer` is set and present in the archive is not
re-downloaded; if it is still in R2 it is eligible for deletion; if R2 is already empty there is
nothing to do. Running twice never loses or duplicates data — the archive_pointer is written only
after a verified copy exists, and R2 deletion happens only after that.

Archive location: server.archive's filesystem backend, rooted at ARCHIVE_BASE. This script
defaults ARCHIVE_BASE to <repo>/submissions_raw (already gitignored) if it isn't set; override with
--archive-dir or the ARCHIVE_BASE env var. Layout under it: {model_id}_val_{strategy}/{filename}.

Requirements (same as the rest of server.*): R2 creds in env — R2_ENDPOINT, R2_BUCKET,
R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY (R2_ENDPOINT must be the EU-jurisdiction endpoint the rest
of the pipeline uses). OBJECTSTORE_BACKEND defaults to r2.
"""

import argparse
import glob
import hashlib
import os
import re
import shutil
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from server import metadata as meta_mod  # noqa: E402
from server.archive import get_archive_backend, submission_key, sha256_of  # noqa: E402
from server.objectstore import get_object_store, incoming_key, ObjectStoreError  # noqa: E402

SUBMISSIONS_DIR = os.path.join(_REPO_ROOT, "submissions")
DEFAULT_ARCHIVE_DIR = os.path.join(_REPO_ROOT, "submissions_raw")
_MD5_RE = re.compile(r"^[0-9a-f]{32}$")


# --------------------------------------------------------------------------- helpers

def human(n):
    """Bytes -> short human string."""
    if n is None:
        return "?"
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024


def file_digests(path, chunk=1 << 20):
    """Single-pass (size, md5_hex, sha256_hex) of a local file."""
    md5, sha, size = hashlib.md5(), hashlib.sha256(), 0
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            md5.update(b)
            sha.update(b)
            size += len(b)
    return size, md5.hexdigest(), sha.hexdigest()


def verify_download(path, r2_size, r2_etag, expected_sha256):
    """Check a freshly downloaded file against R2's metadata and the scoring sha256.

    Returns (problems, note). `problems` empty == verified (safe to archive/delete).
    """
    size, md5_hex, sha_hex = file_digests(path)
    problems, checks = [], [f"size {human(size)} == R2"]
    if size != r2_size:
        problems.append(f"size {size} != R2-reported {r2_size}")
        checks[-1] = f"size {human(size)} != R2 {human(r2_size)}"
    if r2_etag and _MD5_RE.match(r2_etag):
        if md5_hex == r2_etag:
            checks.append("md5 == ETag")
        else:
            problems.append(f"md5 {md5_hex} != ETag {r2_etag}")
            checks.append("md5 != ETag")
    else:
        checks.append("ETag check skipped (multipart/absent)")
    if expected_sha256:
        if sha_hex == expected_sha256:
            checks.append("sha256 == metadata")
        else:
            problems.append(f"sha256 {sha_hex} != metadata {expected_sha256}")
            checks.append("sha256 != metadata")
    else:
        checks.append("no metadata sha256 to compare")
    return problems, "; ".join(checks), sha_hex


def archived_sha256_if_local(pointer):
    """Re-hash the stored copy for a file:// pointer (final pre-delete guard). None if not local."""
    if pointer and pointer.startswith("file://"):
        p = pointer[len("file://"):]
        if os.path.exists(p):
            return sha256_of(p)
    return None


def write_pointer(meta_path, meta, filename, pointer):
    """Set archive_pointer for one file and persist metadata.yaml atomically (mutates `meta`)."""
    for f in meta["files"]:
        if f["filename"] == filename:
            f["archive_pointer"] = pointer
            break
    tmp = meta_path + ".tmp"
    meta_mod.write_metadata(tmp, meta)
    os.replace(tmp, meta_path)


def find_scored(model_id=None, val_strategy=None):
    """All submissions/*/metadata.yaml with status `scored`, optionally filtered. Sorted."""
    out = []
    for path in sorted(glob.glob(os.path.join(SUBMISSIONS_DIR, "*", "metadata.yaml"))):
        try:
            meta = meta_mod.load_metadata(path)
        except Exception as e:
            print(f"  ! {path}: could not read metadata ({e}); skipping", file=sys.stderr)
            continue
        if meta.get("status") != meta_mod.STATUS_SCORED:
            continue
        if model_id and meta.get("model_id") != model_id:
            continue
        if val_strategy and meta.get("val_strategy") != val_strategy:
            continue
        out.append((path, meta))
    return out


# --------------------------------------------------------------------------- sweep

def sweep_submission(meta_path, meta, store, backend, *, do_archive, do_delete, tmp_root, stats):
    """Archive + (optionally) delete one scored submission's files. Returns True if it did/would
    do anything. Records its own counts into `stats`."""
    model_id, val = meta["model_id"], meta["val_strategy"]
    label = f"{model_id}_val_{val}"
    print(f"\n=== {label}  (status: {meta['status']}) ===")
    touched = False

    for f in meta["files"]:
        fn = f["filename"]
        r2_key = f.get("r2_key") or incoming_key(model_id, val, fn)
        key = submission_key(model_id, val, fn)
        expected_sha = f.get("sha256")
        pointer = f.get("archive_pointer")
        archived = bool(pointer) and backend.exists(pointer)

        try:
            in_r2 = store.exists(r2_key)
        except ObjectStoreError as e:
            print(f"  ! {fn}: cannot reach R2 ({e}); skipping")
            stats["errors"] += 1
            continue

        # --- Already archived ------------------------------------------------
        if archived:
            if not in_r2:
                print(f"  ok {fn}: already archived, R2 empty — nothing to do")
                continue
            size = store.head(r2_key)["size"]
            touched = True
            if do_delete:
                store.delete(r2_key)
                print(f"  ✓ {fn}: already archived; DELETED from R2 ({human(size)} freed)")
                stats["deleted"] += 1
                stats["bytes_freed"] += size or 0
            else:
                print(f"  • {fn}: already archived; would DELETE from R2 ({human(size)}) "
                      f"[--confirm]")
                stats["would_delete"] += 1
                stats["bytes_would_free"] += size or 0
            continue

        # --- Not archived: needs the raw file in R2 --------------------------
        if not in_r2:
            if pointer:  # pointer recorded but the archive copy is gone, and R2 is empty too
                print(f"  ! {fn}: archive_pointer set but archived copy MISSING and R2 empty — "
                      f"raw file may be LOST ({pointer})")
            else:
                print(f"  ! {fn}: not in R2 ({r2_key}) and not archived — cannot archive; skipping")
            stats["errors"] += 1
            continue

        head = store.head(r2_key)
        r2_size, r2_etag = head["size"], head["etag"]
        touched = True

        # Preview only.
        if not do_archive:
            etag_note = "md5 vs ETag" if (r2_etag and _MD5_RE.match(r2_etag)) else "ETag skipped"
            print(f"  • {fn}: would download {human(r2_size)} -> archive {key}; "
                  f"verify (size, {etag_note}, sha256 vs metadata); record archive_pointer; "
                  f"then would DELETE from R2 [--confirm]")
            stats["would_archive"] += 1
            stats["bytes_would_archive"] += r2_size or 0
            stats["would_delete"] += 1
            stats["bytes_would_free"] += r2_size or 0
            continue

        # Download -> verify -> store -> guard -> record -> (delete).
        tmpdir = tempfile.mkdtemp(prefix="sweep_", dir=tmp_root)
        tmp = os.path.join(tmpdir, fn)
        try:
            print(f"  … {fn}: downloading {human(r2_size)} from {r2_key}")
            store.get(r2_key, tmp)

            problems, note, dl_sha = verify_download(tmp, r2_size, r2_etag, expected_sha)
            if problems:
                print(f"  ! {fn}: verification FAILED ({'; '.join(problems)}) — "
                      f"NOT archived, NOT deleted")
                stats["errors"] += 1
                continue
            print(f"  ✓ {fn}: verified ({note})")

            pointer = backend.store(tmp, key)
            stored_sha = archived_sha256_if_local(pointer)
            if not backend.exists(pointer) or (stored_sha is not None and stored_sha != dl_sha):
                print(f"  ! {fn}: archived copy failed post-store check — NOT deleting from R2")
                stats["errors"] += 1
                continue

            write_pointer(meta_path, meta, fn, pointer)
            print(f"  ✓ {fn}: archived -> {pointer}; recorded archive_pointer")
            stats["archived"] += 1
            stats["bytes_archived"] += r2_size or 0

            if do_delete:
                store.delete(r2_key)
                print(f"  ✓ {fn}: DELETED from R2 ({human(r2_size)} freed)")
                stats["deleted"] += 1
                stats["bytes_freed"] += r2_size or 0
            else:
                print(f"  • {fn}: archived; left in R2 ({human(r2_size)}) — use --confirm to delete")
                stats["would_delete"] += 1
                stats["bytes_would_free"] += r2_size or 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    if touched:
        stats["submissions"] += 1
    else:
        print("  (nothing to do)")
    return touched


# --------------------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Manual archive sweep: move scored submissions' raw files from R2 to the local archive.")
    ap.add_argument("--confirm", action="store_true",
                    help="archive AND delete each verified file from R2 (default: dry-run preview)")
    ap.add_argument("--download", action="store_true",
                    help="do the archiving (download/verify/store/record) but do NOT delete from R2")
    ap.add_argument("--model-id", default=None, help="limit to this model_id")
    ap.add_argument("--val-strategy", default=None, help="limit to this val_strategy")
    ap.add_argument("--archive-dir", default=None,
                    help=f"archive root (default: $ARCHIVE_BASE or {DEFAULT_ARCHIVE_DIR})")
    args = ap.parse_args(argv)

    # Pin the archive to a local filesystem dir unless the env already configures one.
    if args.archive_dir:
        os.environ["ARCHIVE_BASE"] = os.path.abspath(args.archive_dir)
    os.environ.setdefault("ARCHIVE_BACKEND", "filesystem")
    os.environ.setdefault("ARCHIVE_BASE", DEFAULT_ARCHIVE_DIR)

    do_delete = args.confirm
    do_archive = args.confirm or args.download
    mode = "CONFIRM (archive + delete from R2)" if do_delete else (
        "DOWNLOAD (archive only, no R2 delete)" if do_archive else "DRY RUN (preview only)")

    backend = get_archive_backend()
    store = get_object_store()

    # Local staging area for downloads (same filesystem as the archive when possible).
    tmp_root = None
    if do_archive:
        base = getattr(backend, "base", None)
        tmp_root = os.path.join(base, ".sweep_tmp") if base else tempfile.gettempdir()
        os.makedirs(tmp_root, exist_ok=True)

    subs = find_scored(args.model_id, args.val_strategy)
    print(f"Archive sweep — mode: {mode}")
    print(f"Archive root:   {os.environ.get('ARCHIVE_BASE')} (backend={backend.scheme})")
    print(f"Scored submissions to consider: {len(subs)}")
    if not subs:
        print("Nothing scored matches — nothing to do.")
        return 0

    stats = {k: 0 for k in (
        "submissions", "archived", "deleted", "would_archive", "would_delete",
        "bytes_archived", "bytes_freed", "bytes_would_archive", "bytes_would_free", "errors")}

    for meta_path, meta in subs:
        sweep_submission(meta_path, meta, store, backend,
                         do_archive=do_archive, do_delete=do_delete,
                         tmp_root=tmp_root, stats=stats)

    # --- summary ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Summary ({mode}):")
    print(f"  submissions touched: {stats['submissions']}")
    if do_archive:
        print(f"  files archived:      {stats['archived']}  ({human(stats['bytes_archived'])})")
        if do_delete:
            print(f"  files deleted in R2: {stats['deleted']}  "
                  f"({human(stats['bytes_freed'])} freed)")
        else:
            print(f"  would delete in R2:  {stats['would_delete']}  "
                  f"({human(stats['bytes_would_free'])} reclaimable) — re-run with --confirm")
    else:
        print(f"  would archive:       {stats['would_archive']}  "
              f"({human(stats['bytes_would_archive'])})")
        print(f"  would free from R2:  {stats['would_delete']}  "
              f"({human(stats['bytes_would_free'])}) — re-run with --confirm")
    if stats["errors"]:
        print(f"  ERRORS/warnings:     {stats['errors']}  (see ! lines above)")
    print("Done." if not stats["errors"] else "Done WITH ERRORS — see the ! lines above.")
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
