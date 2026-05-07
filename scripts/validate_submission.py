"""
Validate FLUXNET leaderboard submission files.

Called by the GitHub Actions validate-pr workflow with the list of files
changed in the pull request. Collects ALL errors before exiting so the
submitter sees every problem at once.

Usage:
    python scripts/validate_submission.py path/to/file1.csv path/to/file2.csv \\
        [--base-ref origin/main]

Exit codes:
    0 — all files valid and completeness rules satisfied
    1 — one or more errors found
"""

import argparse
import os
import subprocess
import sys

import pandas as pd

VALID_SETTINGS = {'time-split', 'spatial-easy40', 'TA40'}
VALID_TARGETS = {'GPP', 'ET', 'NEE'}
VALID_VAL_STRATEGIES = {'mean', 'max', 'discrepancy'}

REQUIRED_COLUMNS = [
    'target', 'setting', 'model', 'scale', 'env', 'n_samples',
    'mse', 'rmse', 'mae', 'nse', 'r2_score', 'bias', 'relative_mae', 'relative_bias'
]

FULL_SET = {
    (setting, target)
    for setting in VALID_SETTINGS
    for target in VALID_TARGETS
}


def parse_filename(filepath):
    """
    Parse a submission filepath.

    Expected: submissions/{model_name}/{setting}_{target}_{model_name}_val_{val_strategy}.csv

    Returns dict with keys: model_folder, setting, target, model_name, val_strategy
    Or None if the path/filename doesn't match.
    """
    parts = filepath.replace('\\', '/').split('/')
    # Find 'submissions' in path
    try:
        sub_idx = parts.index('submissions')
    except ValueError:
        return None

    relative_parts = parts[sub_idx:]
    if len(relative_parts) != 3:
        return None
    _, model_folder, filename = relative_parts

    if not filename.endswith('.csv'):
        return None
    base = filename[:-4]

    val_strategy = None
    for strategy in VALID_VAL_STRATEGIES:
        suffix = f'_val_{strategy}'
        if base.endswith(suffix):
            val_strategy = strategy
            base = base[:-len(suffix)]
            break
    if val_strategy is None:
        return None

    setting = None
    for s in sorted(VALID_SETTINGS, key=len, reverse=True):
        if base.startswith(f'{s}_'):
            setting = s
            base = base[len(f'{s}_'):]
            break
    if setting is None:
        return None

    target = None
    for t in VALID_TARGETS:
        if base.startswith(f'{t}_'):
            target = t
            model_name = base[len(f'{t}_'):]
            break
    if target is None:
        return None

    return {
        'model_folder': model_folder,
        'setting': setting,
        'target': target,
        'model_name': model_name,
        'val_strategy': val_strategy,
        'filename': filename,
        'filepath': filepath,
    }


def get_files_on_base(base_ref, path_prefix='submissions/'):
    """Return the set of files that exist on the base branch under path_prefix."""
    try:
        result = subprocess.run(
            ['git', 'ls-tree', '-r', '--name-only', base_ref, path_prefix],
            capture_output=True, text=True, check=True
        )
        return set(result.stdout.strip().split('\n')) - {''}
    except subprocess.CalledProcessError:
        return set()


def validate_files(changed_files, base_ref=None):
    """
    Validate all changed files. Returns (errors, warnings, parsed_records).
    """
    errors = []
    warnings = []
    parsed_records = []

    files_on_base = get_files_on_base(base_ref) if base_ref else set()

    for filepath in changed_files:
        filepath = filepath.strip()
        if not filepath:
            continue

        # Rule: only submissions/ files allowed
        norm = filepath.replace('\\', '/')
        if not norm.startswith('submissions/'):
            errors.append(
                f"**{filepath}**: PRs may only add files under `submissions/`. "
                "Changes to workflows, scripts, or source code must be opened by a maintainer."
            )
            continue

        parsed = parse_filename(filepath)
        if parsed is None:
            errors.append(
                f"**{filepath}**: Filename does not match the required pattern "
                "`submissions/{{model_name}}/{{setting}}_{{target}}_{{model_name}}_val_{{val_strategy}}.csv`.\n"
                f"  - Valid settings: {sorted(VALID_SETTINGS)}\n"
                f"  - Valid targets: {sorted(VALID_TARGETS)}\n"
                f"  - Valid val_strategies: {sorted(VALID_VAL_STRATEGIES)}"
            )
            continue

        model_folder = parsed['model_folder']
        model_name = parsed['model_name']
        setting = parsed['setting']
        target = parsed['target']

        # Rule: folder name must match model_name in filename
        if model_folder != model_name:
            errors.append(
                f"**{filepath}**: Folder name `{model_folder}` does not match "
                f"model name `{model_name}` in the filename."
            )
            continue

        # Load and validate CSV content
        if not os.path.exists(filepath):
            errors.append(f"**{filepath}**: File not found on disk.")
            continue

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            errors.append(f"**{filepath}**: Could not read CSV — {e}")
            continue

        file_errors = []

        # Check columns
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        extra_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]
        if missing_cols:
            file_errors.append(f"  - Missing columns: {missing_cols}")
        if extra_cols:
            warnings.append(f"**{filepath}**: Extra columns (ignored): {extra_cols}")

        if not missing_cols:
            # Check model column matches folder/filename
            model_vals = df['model'].dropna().unique().tolist()
            if model_vals != [model_name]:
                file_errors.append(
                    f"  - `model` column must contain exactly `{model_name}`, "
                    f"found: {model_vals}"
                )

            # Check setting column
            setting_vals = df['setting'].dropna().unique().tolist()
            if setting_vals != [setting]:
                file_errors.append(
                    f"  - `setting` column must contain exactly `{setting}`, "
                    f"found: {setting_vals}"
                )

            # Check target column
            target_vals = df['target'].dropna().unique().tolist()
            if target_vals != [target]:
                file_errors.append(
                    f"  - `target` column must contain exactly `{target}`, "
                    f"found: {target_vals}"
                )

            # n_samples must never be NaN; metric NaNs are expected for some scales
            if 'n_samples' in df.columns and df['n_samples'].isna().any():
                n_nan = df['n_samples'].isna().sum()
                file_errors.append(f"  - Column `n_samples` has {n_nan} NaN value(s).")

        if file_errors:
            errors.append(f"**{filepath}**:\n" + "\n".join(file_errors))
        else:
            parsed['_df_ok'] = True
            parsed_records.append(parsed)

    return errors, warnings, parsed_records, files_on_base


def check_completeness(parsed_records, files_on_base):
    """
    Check the 9-file completeness rule.

    Groups changed files by (model_name, val_strategy). For each group:
    - If it's a brand-new val_strategy (no files exist on base), all 9 must be present.
    - If updating an existing val_strategy (all 9 already on base), any subset is OK.
    - Mixed state (some but not all 9 on base) is always an error.
    """
    errors = []

    groups = {}
    for r in parsed_records:
        key = (r['model_name'], r['val_strategy'])
        groups.setdefault(key, []).append(r)

    for (model_name, val_strategy), records in groups.items():
        changed_combos = {(r['setting'], r['target']) for r in records}

        # Check for duplicate (setting, target) within this group
        seen = {}
        for r in records:
            combo = (r['setting'], r['target'])
            if combo in seen:
                errors.append(
                    f"**{model_name} / {val_strategy}**: Duplicate file for "
                    f"setting=`{combo[0]}`, target=`{combo[1]}`."
                )
            seen[combo] = r

        # Count how many of the 9 already exist on base branch
        existing_on_base = set()
        for setting, target in FULL_SET:
            expected_filename = (
                f"submissions/{model_name}/"
                f"{setting}_{target}_{model_name}_val_{val_strategy}.csv"
            )
            if expected_filename in files_on_base:
                existing_on_base.add((setting, target))

        n_existing = len(existing_on_base)

        if n_existing == 0:
            # Fresh submission — all 9 required
            missing = FULL_SET - changed_combos
            if missing:
                missing_list = sorted(f"`{s}` × `{t}`" for s, t in missing)
                errors.append(
                    f"**{model_name} / {val_strategy}**: New submission must include "
                    f"all 9 files (3 settings × 3 targets). Missing:\n"
                    + "\n".join(f"  - {m}" for m in missing_list)
                )
        elif n_existing == 9:
            # Full update — any subset is fine; just ensure no unknown combos
            unknown = changed_combos - FULL_SET
            if unknown:
                errors.append(
                    f"**{model_name} / {val_strategy}**: Unknown (setting, target) "
                    f"combinations: {sorted(unknown)}"
                )
        else:
            # Partial state on base — this shouldn't normally happen
            errors.append(
                f"**{model_name} / {val_strategy}**: The base branch has {n_existing}/9 "
                f"files for this val_strategy, which is an inconsistent state. "
                f"Please ensure the existing submission on `main` is complete before updating."
            )

    return errors


def build_report(file_errors, completeness_errors, warnings, parsed_records):
    """Build the markdown-formatted report string."""
    n_ok = len(parsed_records)
    n_err = len(file_errors) + len(completeness_errors)

    lines = []

    if n_err == 0:
        lines.append("## Submission validation: PASSED")
        lines.append(f"\n{n_ok} file(s) validated successfully.")
    else:
        lines.append("## Submission validation: FAILED")
        lines.append(f"\n{n_err} error(s) found across {n_ok + n_err} file(s).\n")

    if file_errors:
        lines.append("### File errors\n")
        for e in file_errors:
            lines.append(f"- {e}\n")

    if completeness_errors:
        lines.append("### Completeness errors\n")
        for e in completeness_errors:
            lines.append(f"- {e}\n")

    if warnings:
        lines.append("### Warnings (non-blocking)\n")
        for w in warnings:
            lines.append(f"- {w}\n")

    if n_ok > 0 and n_err == 0:
        lines.append("\n### Valid files\n")
        for r in parsed_records:
            lines.append(
                f"- `{r['filepath']}` — model=`{r['model_name']}`, "
                f"setting=`{r['setting']}`, target=`{r['target']}`, "
                f"val_strategy=`{r['val_strategy']}`"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Validate FLUXNET leaderboard submissions")
    parser.add_argument('files', nargs='*', help="Changed file paths to validate")
    parser.add_argument('--base-ref', default=None,
                        help="Base git ref to check for existing files (e.g. origin/main)")
    args = parser.parse_args()

    changed_files = args.files
    if not changed_files:
        print("## Submission validation: SKIPPED\n\nNo changed files provided.")
        sys.exit(0)

    # Filter to only submission files (skip if workflow/scripts accidentally passed)
    submission_files = [f for f in changed_files if 'submissions/' in f.replace('\\', '/')]
    non_submission = [f for f in changed_files if 'submissions/' not in f.replace('\\', '/')]

    file_errors = []
    if non_submission:
        for f in non_submission:
            file_errors.append(
                f"**{f}**: PRs may only modify files under `submissions/`. "
                "Changes to other files require a separate maintainer PR."
            )

    file_errs, warnings, parsed_records, files_on_base = validate_files(
        submission_files, base_ref=args.base_ref
    )
    file_errors.extend(file_errs)

    completeness_errors = check_completeness(parsed_records, files_on_base)

    report = build_report(file_errors, completeness_errors, warnings, parsed_records)
    print(report)

    if file_errors or completeness_errors:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
