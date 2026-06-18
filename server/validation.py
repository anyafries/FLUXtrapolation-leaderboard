"""
Standalone submission validator — the single source of truth for "is this submission valid?".

Called identically from the GitHub Action, the relay, and the VM (defensive re-validation).
None of the logic lives in workflow YAML; all callers import `validate_submission`.

Five checks (all run; the report lists every failure at once):
  1. files            — exactly the 9 expected raw files (3 settings x 3 targets) for one
                        (model_id, val_strategy), with correct filenames.
  2. content_caps     — per file: size cap, required columns, unique (site_id,time),
                        y_pred finite + in magnitude range, no NaNs; bounded-time parse;
                        data only (CSV is never executed/eval'd).
  3. index_completeness — the submission's (site_id,time) set EXACTLY matches the truth index
                        for that (setting,target). Exact match (not count/site) because the
                        aggregations average over whatever rows are present, so cherry-picking
                        sub-rows changes scores while leaving coarse counts unchanged.
  4. ytrue_integrity  — submitted y_true matches the lr truth within tolerance on (site_id,time).
                        Scoring always uses lr's values; this just rejects wrong-data uploads.
  5. ownership        — if the model_id already exists, the submitter's owner token must match.

Truth-table access goes through server.truth (fetch+cache+verify sha256; partition-only load).

Output: a ValidationReport with a machine-readable dict/JSON (gates auto-merge) and a markdown
rendering (for the PR comment).
"""

import json
import os
import signal
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from server import truth as truth_mod  # noqa: E402

VALID_SETTINGS = {"time-split", "spatial-easy40", "TA40"}
VALID_TARGETS = {"GPP", "ET", "NEE"}
VALID_VAL_STRATEGIES = {"mean", "max", "discrepancy"}
FULL_SET = {(s, t) for s in VALID_SETTINGS for t in VALID_TARGETS}

RAW_REQUIRED_COLUMNS = ["y_true", "y_pred", "env", "site_id", "time"]


@dataclass
class ValidationConfig:
    max_file_bytes: int = 400_000_000        # > the largest real file (~255 MB); DoS guard
    max_abs_ypred: float = 1e6               # sanity bound (not physical); finite is required
    ytrue_rtol: float = 1e-5                 # y_true must match lr truth within this
    ytrue_atol: float = 1e-6
    ytrue_max_mismatch_frac: float = 0.0     # fraction of rows allowed out of tolerance
    parse_timeout_s: int = 180               # bounded-time parse (best-effort; size cap backs it)


@dataclass
class CheckResult:
    name: str
    passed: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def fail(self, msg):
        self.passed = False
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


@dataclass
class ValidationReport:
    model_id: str
    val_strategy: str
    checks: list  # list[CheckResult]

    @property
    def passed(self):
        return all(c.passed for c in self.checks)

    def to_dict(self):
        return {
            "passed": self.passed,
            "model_id": self.model_id,
            "val_strategy": self.val_strategy,
            "checks": [
                {"name": c.name, "passed": c.passed, "errors": c.errors, "warnings": c.warnings}
                for c in self.checks
            ],
        }

    def to_json(self, **kw):
        return json.dumps(self.to_dict(), **kw)

    def to_markdown(self):
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"## Submission validation: {status}",
                 f"\n`{self.model_id}` / `val_{self.val_strategy}`\n"]
        for c in self.checks:
            mark = "✅" if c.passed else "❌"
            lines.append(f"- {mark} **{c.name}**")
            for e in c.errors:
                lines.append(f"    - {e}")
            for w in c.warnings:
                lines.append(f"    - ⚠️ {w}")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Filename parsing
# --------------------------------------------------------------------------------------

def parse_raw_filename(path):
    """Parse {setting}_{target}_{model}_val_{strategy}[_predictions].csv.

    Returns dict(setting, target, model, val_strategy, filename) or None.
    """
    filename = os.path.basename(path)
    if not filename.endswith(".csv"):
        return None
    base = filename[:-4]
    if base.endswith("_predictions"):
        base = base[: -len("_predictions")]

    val_strategy = None
    for s in VALID_VAL_STRATEGIES:
        if base.endswith(f"_val_{s}"):
            val_strategy = s
            base = base[: -len(f"_val_{s}")]
            break
    if val_strategy is None:
        return None

    setting = None
    for s in sorted(VALID_SETTINGS, key=len, reverse=True):
        if base.startswith(f"{s}_"):
            setting = s
            base = base[len(f"{s}_"):]
            break
    if setting is None:
        return None

    for t in VALID_TARGETS:
        if base.startswith(f"{t}_"):
            return {"setting": setting, "target": t, "model": base[len(f"{t}_"):],
                    "val_strategy": val_strategy, "filename": filename}
    return None


# --------------------------------------------------------------------------------------
# Bounded-time, data-only CSV read
# --------------------------------------------------------------------------------------

class _ParseTimeout(Exception):
    pass


@contextmanager
def _time_limit(seconds):
    """Best-effort wall-clock limit via SIGALRM (main thread + Unix only)."""
    usable = (seconds and seconds > 0
              and hasattr(signal, "SIGALRM")
              and threading.current_thread() is threading.main_thread())
    if not usable:
        yield
        return

    def _handler(signum, frame):
        raise _ParseTimeout(f"parse exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _read_predictions(path, config):
    """Read a raw prediction CSV as data only (C engine, explicit dtypes); never executes content."""
    with _time_limit(config.parse_timeout_s):
        header = pd.read_csv(path, nrows=0, engine="c")
        missing = [c for c in RAW_REQUIRED_COLUMNS if c not in header.columns]
        if missing:
            raise ValueError(f"missing columns {missing}")
        df = pd.read_csv(
            path, engine="c", usecols=RAW_REQUIRED_COLUMNS,
            dtype={"y_true": "float64", "y_pred": "float64", "env": "string", "site_id": "string"},
        )
    df["site_id"] = df["site_id"].astype(str)
    df["time"] = pd.to_datetime(df["time"])
    return df


# --------------------------------------------------------------------------------------
# The five checks
# --------------------------------------------------------------------------------------

def _check_files(files, model_id, val_strategy):
    """CHECK 1. Returns (CheckResult, parsed: dict[(setting,target)->path])."""
    c = CheckResult("files")
    parsed = {}
    seen = {}
    for path in files:
        meta = parse_raw_filename(path)
        if meta is None:
            c.fail(f"`{os.path.basename(path)}`: filename does not match "
                   "`{setting}_{target}_{model}_val_{strategy}_predictions.csv`")
            continue
        if meta["model"] != model_id:
            c.fail(f"`{meta['filename']}`: model `{meta['model']}` != claimed `{model_id}`")
            continue
        if meta["val_strategy"] != val_strategy:
            c.fail(f"`{meta['filename']}`: val_strategy `{meta['val_strategy']}` != claimed `{val_strategy}`")
            continue
        combo = (meta["setting"], meta["target"])
        if combo in seen:
            c.fail(f"duplicate file for {combo[0]} x {combo[1]}: "
                   f"`{seen[combo]}` and `{meta['filename']}`")
            continue
        seen[combo] = meta["filename"]
        parsed[combo] = path

    missing = FULL_SET - set(parsed)
    if missing:
        c.fail("missing files for: " + ", ".join(sorted(f"{s} x {t}" for s, t in missing)))
    return c, parsed


def _check_content(df, setting, target, config, c):
    """CHECK 2 (per file): structural caps. Mutates CheckResult `c`; returns ok bool."""
    tag = f"{setting} x {target}"
    n_dup = int(df.duplicated(["site_id", "time"]).sum())
    if n_dup:
        c.fail(f"{tag}: {n_dup} duplicate (site_id, time) rows")

    yp = df["y_pred"].to_numpy(float)
    n_nan = int(np.sum(np.isnan(yp)))
    n_inf = int(np.sum(np.isinf(yp)))
    if n_nan:
        c.fail(f"{tag}: {n_nan} NaN values in y_pred")
    if n_inf:
        c.fail(f"{tag}: {n_inf} non-finite (inf) values in y_pred")
    n_big = int(np.sum(np.abs(yp) > config.max_abs_ypred))
    if n_big:
        c.fail(f"{tag}: {n_big} y_pred values exceed magnitude cap {config.max_abs_ypred:g}")
    return n_dup == 0 and n_nan == 0 and n_inf == 0


def _check_index(df, truth_df, setting, target, config, c, n_examples=5):
    """CHECK 3 (per file): exact (site_id,time) set match against the truth index."""
    tag = f"{setting} x {target}"
    sub = df[["site_id", "time"]].drop_duplicates()
    req = truth_df[["site_id", "time"]]
    merged = sub.merge(req, on=["site_id", "time"], how="outer", indicator=True)
    extra = merged[merged["_merge"] == "left_only"]
    missing = merged[merged["_merge"] == "right_only"]
    if len(missing):
        ex = ", ".join(f"({r.site_id} @ {r.time})" for r in missing.head(n_examples).itertuples())
        c.fail(f"{tag}: {len(missing)} required rows missing (e.g. {ex})")
    if len(extra):
        ex = ", ".join(f"({r.site_id} @ {r.time})" for r in extra.head(n_examples).itertuples())
        c.fail(f"{tag}: {len(extra)} rows not in the canonical index (e.g. {ex})")


def _check_integrity(df, truth_df, setting, target, config, c):
    """CHECK 4 (per file): submitted y_true matches lr truth within tolerance."""
    tag = f"{setting} x {target}"
    m = df[["site_id", "time", "y_true"]].merge(
        truth_df[["site_id", "time", "y_true"]], on=["site_id", "time"], suffixes=("_sub", "_truth")
    )
    if m.empty:
        c.fail(f"{tag}: no overlap with truth index for integrity check")
        return
    a = m["y_true_sub"].to_numpy(float)
    b = m["y_true_truth"].to_numpy(float)
    bad = ~np.isclose(a, b, rtol=config.ytrue_rtol, atol=config.ytrue_atol, equal_nan=False)
    n_bad = int(np.sum(bad))
    frac = n_bad / len(m)
    if frac > config.ytrue_max_mismatch_frac:
        worst = float(np.nanmax(np.abs(a[bad] - b[bad]))) if n_bad else 0.0
        c.fail(f"{tag}: submitted y_true disagrees with lr truth on {n_bad}/{len(m)} rows "
               f"({frac:.2%}; max |Δ|={worst:.3g}) — wrong or mismatched data")


def _check_ownership(model_id, owner_token, recorded_owner):
    """CHECK 5. New model_id claims ownership; existing requires matching owner token."""
    c = CheckResult("ownership")
    if recorded_owner is None:
        if not owner_token:
            c.warn(f"new model_id `{model_id}` with no owner token recorded")
        return c
    if owner_token != recorded_owner:
        c.fail(f"model_id `{model_id}` is owned by another submitter; owner token does not match")
    return c


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def validate_submission(files, *, model_id, val_strategy, owner_token=None, recorded_owner=None,
                        truth_source=None, truth_path=None, manifest_path=truth_mod.DEFAULT_MANIFEST,
                        config=None):
    """Validate one submission (its 9 raw prediction files). Returns a ValidationReport.

    Either pass `truth_path` (a locally verified parquet) or `truth_source`/$TRUTH_TABLE_URL,
    in which case the truth table is fetched, cached, and sha256-verified here.
    """
    config = config or ValidationConfig()
    if truth_path is None:
        truth_path = truth_mod.ensure_local_truth(truth_source, manifest_path=manifest_path)

    c_files, parsed = _check_files(files, model_id, val_strategy)
    c_content = CheckResult("content_caps")
    c_index = CheckResult("index_completeness")
    c_integrity = CheckResult("ytrue_integrity")

    for (setting, target), path in sorted(parsed.items()):
        size = os.path.getsize(path)
        if size > config.max_file_bytes:
            c_content.fail(f"{setting} x {target}: file is {size} bytes (> cap {config.max_file_bytes})")
            continue
        try:
            df = _read_predictions(path, config)
        except _ParseTimeout as e:
            c_content.fail(f"{setting} x {target}: {e}")
            continue
        except ValueError as e:
            c_content.fail(f"{setting} x {target}: {e}")
            continue

        structural_ok = _check_content(df, setting, target, config, c_content)
        truth_df = truth_mod.load_partition(setting, target, truth_path)
        _check_index(df, truth_df, setting, target, config, c_index)
        if structural_ok:
            _check_integrity(df, truth_df, setting, target, config, c_integrity)

    c_owner = _check_ownership(model_id, owner_token, recorded_owner)
    checks = [c_files, c_content, c_index, c_integrity, c_owner]
    return ValidationReport(model_id=model_id, val_strategy=val_strategy, checks=checks)


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Validate a submission's raw prediction files")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--val-strategy", required=True, choices=sorted(VALID_VAL_STRATEGIES))
    ap.add_argument("--owner-token", default=None)
    ap.add_argument("--recorded-owner", default=None)
    ap.add_argument("--truth-source", default=None, help="local path or http(s)/file URL")
    ap.add_argument("--truth-path", default=None, help="pre-verified local truth parquet")
    ap.add_argument("--json-out", default=None, help="write machine-readable report here")
    args = ap.parse_args()

    report = validate_submission(
        args.files, model_id=args.model_id, val_strategy=args.val_strategy,
        owner_token=args.owner_token, recorded_owner=args.recorded_owner,
        truth_source=args.truth_source, truth_path=args.truth_path,
    )
    print(report.to_markdown())
    if args.json_out:
        with open(args.json_out, "w") as f:
            f.write(report.to_json(indent=2))
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    _cli()
