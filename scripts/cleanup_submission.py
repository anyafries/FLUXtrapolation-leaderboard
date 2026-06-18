#!/usr/bin/env python3
"""
Maintainer cleanup: fully remove one submission (model_id + val_strategy).

Removes, in order:
  1. R2 transient objects under  incoming/{model_id}_val_{strategy}/   (server.objectstore)
  2. the Cloudflare KV ownership key  owner:{model_id}                 (via wrangler: local + remote)
  3. the repo folder  submissions/{model_id}_val_{strategy}/           (git rm, or rmtree if untracked)

DRY RUN by default: it prints exactly what it WOULD delete and changes nothing.
Pass --confirm to actually delete.

    python scripts/cleanup_submission.py --model-id foo --val-strategy mean            # preview
    python scripts/cleanup_submission.py --model-id foo --val-strategy mean --confirm  # delete

Footguns this script guards against:
  - Ownership (owner:{model_id}) is per MODEL, not per val_strategy. Deleting it unclaims the
    model for ALL of its val_strategy submissions — anyone could then re-register the id. If
    other submissions/{model_id}_val_*/ folders remain, the script warns; use --keep-owner to
    leave the id claimed and only drop this strategy's R2 objects + folder.

Requirements (same as the rest of server.*): R2 creds in env (R2_ENDPOINT/R2_BUCKET/
R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY), and a configured `wrangler` for the KV step.

# TODO(Phase 4) — keep-forever ARCHIVE is intentionally NOT touched here.
# Once the VM (scripts/process_submission.py) archives raw predictions via server.archive and
# records `archive_pointer` per file in metadata.yaml (status: scored), this cleanup becomes
# INCOMPLETE: it drops the transient R2 copy, the KV claim and the repo folder, but leaves the
# archived copy behind. Decide the policy then, and wire it in here:
#   - PURGE: for each file read `archive_pointer` from metadata.yaml and delete it via the
#            ArchiveBackend (needs a delete() on server.archive.ArchiveBackend — doesn't exist yet), OR
#   - RETAIN: keep the archive as an immutable record of what was scored, and only unlink it here.
# Until decided, the script DETECTS archive_pointer values in metadata.yaml and warns the
# maintainer to handle the archived copy by hand.
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from server.metadata import load_metadata  # noqa: E402
from server.objectstore import get_object_store, ObjectStoreError  # noqa: E402

VALID_STRATEGIES = ("mean", "max", "discrepancy")
MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,48}$")  # mirrors the Worker

WORKER_DIR = os.path.join(_REPO_ROOT, "worker")


# --------------------------------------------------------------------------- helpers

def _run(cmd, **kw):
    """Run a subprocess, returning CompletedProcess; never raises on non-zero."""
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _resolve_wrangler(explicit=None):
    """Find the wrangler binary: --wrangler, then PATH, then `npm config get prefix`/bin."""
    if explicit:
        return explicit
    found = shutil.which("wrangler")
    if found:
        return found
    npm = shutil.which("npm")
    if npm:
        prefix = _run([npm, "config", "get", "prefix"]).stdout.strip()
        cand = os.path.join(prefix, "bin", "wrangler")
        if os.path.exists(cand):
            return cand
    return None


def _git_tracked(relpath):
    """True if `relpath` (repo-relative) has any tracked files."""
    out = _run(["git", "-C", _REPO_ROOT, "ls-files", "--", relpath]).stdout
    return bool(out.strip())


# --------------------------------------------------------------------------- planning

def gather(model_id, val_strategy):
    """Collect everything we'd touch, without changing anything."""
    folder_rel = f"submissions/{model_id}_val_{val_strategy}"
    folder_abs = os.path.join(_REPO_ROOT, folder_rel)
    r2_prefix = f"incoming/{model_id}_val_{val_strategy}/"

    plan = {
        "model_id": model_id,
        "val_strategy": val_strategy,
        "folder_rel": folder_rel,
        "folder_abs": folder_abs,
        "folder_exists": os.path.isdir(folder_abs),
        "folder_tracked": _git_tracked(folder_rel),
        "folder_files": [],
        "r2_prefix": r2_prefix,
        "r2_keys": [],
        "r2_error": None,
        "kv_key": f"owner:{model_id}",
        "metadata": None,
        "archive_pointers": [],
        "siblings": [],
    }

    # Local folder contents.
    if plan["folder_exists"]:
        for dp, _, files in os.walk(folder_abs):
            for fn in files:
                plan["folder_files"].append(
                    os.path.relpath(os.path.join(dp, fn), _REPO_ROOT))
        plan["folder_files"].sort()
        meta_path = os.path.join(folder_abs, "metadata.yaml")
        if os.path.exists(meta_path):
            try:
                meta = load_metadata(meta_path)
                plan["metadata"] = {"status": meta.get("status"), "owner": meta.get("owner")}
                plan["archive_pointers"] = [
                    (f.get("filename"), f.get("archive_pointer"))
                    for f in meta.get("files", []) if f.get("archive_pointer")
                ]
            except Exception as e:  # malformed yaml shouldn't block a cleanup
                plan["metadata"] = {"error": str(e)}

    # Other val_strategy folders for the same model_id (ownership scope warning).
    for p in sorted(glob.glob(os.path.join(_REPO_ROOT, "submissions", f"{model_id}_val_*"))):
        rel = os.path.relpath(p, _REPO_ROOT)
        if rel != folder_rel and os.path.isdir(p):
            plan["siblings"].append(rel)

    # R2 objects (needs creds even for a preview — listing is read-only).
    try:
        plan["r2_keys"] = get_object_store().list_prefix(r2_prefix)
    except ObjectStoreError as e:
        plan["r2_error"] = str(e)

    return plan


def print_plan(plan, keep_owner):
    m, s = plan["model_id"], plan["val_strategy"]
    print(f"\nCleanup target: model_id={m!r}  val_strategy={s!r}\n")

    if plan["metadata"] and "status" in plan["metadata"]:
        print(f"  metadata.yaml: status={plan['metadata']['status']!r} "
              f"owner={plan['metadata']['owner']!r}")
    print()

    # 1) R2
    print(f"[1] R2 objects under  {plan['r2_prefix']}")
    if plan["r2_error"]:
        print(f"    ! could not list R2: {plan['r2_error']}")
    elif plan["r2_keys"]:
        for k in plan["r2_keys"]:
            print(f"    - {k}")
    else:
        print("    (none — already empty; normal for an already-scored submission)")
    print()

    # 2) KV
    print(f"[2] KV key  {plan['kv_key']}  (per-MODEL ownership claim)")
    if keep_owner:
        print("    SKIPPED (--keep-owner): the model_id stays claimed.")
    else:
        print("    will delete from BOTH local (wrangler dev state) and remote (production) KV.")
        if plan["siblings"]:
            print("    !! WARNING: other val_strategy folders for this model_id remain:")
            for sib in plan["siblings"]:
                print(f"         {sib}")
            print("    !! Deleting this key UNCLAIMS the model for those too (anyone could re-register it).")
            print("    !! Use --keep-owner if you only meant to remove this one val_strategy.")
        else:
            print("    (no other val_strategy folders for this model_id; safe to unclaim)")
    print()

    # 3) repo folder
    print(f"[3] repo folder  {plan['folder_rel']}/")
    if not plan["folder_exists"]:
        print("    (does not exist)")
    else:
        how = "git rm -r" if plan["folder_tracked"] else "rmtree (untracked)"
        print(f"    will remove {len(plan['folder_files'])} file(s) via {how}:")
        for f in plan["folder_files"]:
            print(f"    - {f}")
    print()

    # Phase-4 archive warning (see module TODO).
    if plan["archive_pointers"]:
        print("[archive] metadata.yaml records archive_pointer(s) — NOT removed by this script:")
        for fn, ptr in plan["archive_pointers"]:
            print(f"    - {fn}: {ptr}")
        print("    Handle the keep-forever archive copy manually (see TODO(Phase 4) in this script).")
        print()


# --------------------------------------------------------------------------- execution

def execute(plan, keep_owner, wrangler):
    print(">>> --confirm given: DELETING.\n")

    # 1) R2
    if plan["r2_error"]:
        print(f"[1] R2: cannot proceed — listing failed earlier: {plan['r2_error']}")
        print("    Fix R2 creds and re-run; aborting before any deletion.")
        return 1
    store = get_object_store()
    for k in plan["r2_keys"]:
        store.delete(k)
        print(f"[1] R2 deleted: {k}")
    if not plan["r2_keys"]:
        print("[1] R2: nothing to delete.")

    # 2) KV — delete the per-model ownership claim from BOTH the local (wrangler dev) state and
    #    the remote (production) namespace, so a key created during bring-up testing or by the
    #    deployed Worker is gone either way. Each target is handled independently (one missing or
    #    failing must not stop the other); `input="y\n"` answers wrangler's confirmation prompt.
    if keep_owner:
        print(f"[2] KV: skipped (--keep-owner); {plan['kv_key']} left in place.")
    elif wrangler is None:
        print("[2] KV: wrangler not found — could not delete "
              f"{plan['kv_key']}. Install/locate wrangler or use --wrangler PATH, then delete it "
              "manually (or via the Cloudflare dashboard).")
    else:
        for target in ("--local", "--remote"):
            cmd = [wrangler, "kv", "key", "delete", plan["kv_key"], "--binding", "RL", target]
            print(f"[2] KV: $ {' '.join(cmd)}   (cwd={WORKER_DIR})")
            res = _run(cmd, cwd=WORKER_DIR, input="y\n")
            sys.stdout.write(res.stdout)
            sys.stderr.write(res.stderr)
            if res.returncode != 0:
                print(f"[2] KV ({target}): wrangler exited {res.returncode}; "
                      "delete it manually if it still exists.")
            else:
                print(f"[2] KV deleted ({target}): {plan['kv_key']}")

    # 3) repo folder
    if not plan["folder_exists"]:
        print("[3] folder: nothing to remove.")
    elif plan["folder_tracked"]:
        res = _run(["git", "-C", _REPO_ROOT, "rm", "-r", "--", plan["folder_rel"]])
        sys.stdout.write(res.stdout)
        sys.stderr.write(res.stderr)
        if res.returncode == 0:
            print(f"[3] folder: git-removed {plan['folder_rel']}/ (staged — commit to finish).")
        else:
            print(f"[3] folder: git rm exited {res.returncode}.")
            return 1
    else:
        shutil.rmtree(plan["folder_abs"])
        print(f"[3] folder: rmtree'd untracked {plan['folder_rel']}/")

    if plan["archive_pointers"] and not keep_owner:
        print("\n[archive] Reminder: the keep-forever archive copy was NOT removed "
              "(see TODO(Phase 4)).")
    print("\nDone.")
    return 0


# --------------------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(description="Fully remove one submission (model_id + val_strategy).")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--val-strategy", required=True, choices=VALID_STRATEGIES)
    ap.add_argument("--confirm", action="store_true",
                    help="actually delete (default is a dry-run preview)")
    ap.add_argument("--keep-owner", action="store_true",
                    help="do NOT delete owner:{model_id} (keep the model_id claimed)")
    ap.add_argument("--wrangler", default=None,
                    help="path to the wrangler binary (default: PATH, then npm prefix)")
    args = ap.parse_args(argv)

    if not MODEL_ID_RE.match(args.model_id):
        ap.error(f"invalid --model-id {args.model_id!r} (lowercase slug: {MODEL_ID_RE.pattern})")

    plan = gather(args.model_id, args.val_strategy)
    print_plan(plan, args.keep_owner)

    if not args.confirm:
        print("DRY RUN — nothing was deleted. Re-run with --confirm to delete the above.")
        return 0

    wrangler = None if args.keep_owner else _resolve_wrangler(args.wrangler)
    return execute(plan, args.keep_owner, wrangler)


if __name__ == "__main__":
    sys.exit(main())
