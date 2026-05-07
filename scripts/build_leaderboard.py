"""
Build the FLUXNET leaderboard HTML pages from submission CSVs.

Reads all metric CSVs under submissions/{model_name}/ and generates:
  docs/leaderboard_{target}.html         — median RMSE leaderboard
  docs/leaderboard_q90_{target}.html     — 90th-percentile RMSE leaderboard
  docs/index.html                        — landing page linking all leaderboards

Note: This script directly scans submissions/ rather than using eval.py's
load_all_metrics(), because the existing eval.py depends on dataloader.py and
utils/aggregation.py for recomputing metrics from raw predictions. Since
submitters provide pre-computed metric CSVs, no recomputation is needed.
"""

import os
import sys
import re
import pandas as pd

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.plots import create_html_leaderboard
from utils.utils import setup_logging

logger = setup_logging(__name__)

SUBMISSIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'submissions')
DOCS_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs')

VALID_SETTINGS = {'time-split', 'spatial-easy40', 'TA40'}
VALID_TARGETS = {'GPP', 'ET', 'NEE'}
VALID_VAL_STRATEGIES = {'mean', 'max', 'discrepancy'}

DISPLAY_NAMES = {
    "time-split": "temporal",
    "spatial-easy40": "spatial",
    "TA40": "temperature",
}

DROP_SCALES = {'daily', 'monthly'}

REQUIRED_COLUMNS = [
    'target', 'setting', 'model', 'scale', 'env', 'n_samples',
    'mse', 'rmse', 'mae', 'nse', 'r2_score', 'bias', 'relative_mae', 'relative_bias'
]


def parse_submission_filename(filename):
    """
    Parse a submission filename into its components.

    Expected format: {setting}_{target}_{model_name}_val_{val_strategy}.csv

    Returns (setting, target, model_name, val_strategy) or None if unparseable.
    """
    if not filename.endswith('.csv'):
        return None
    base = filename[:-4]

    for strategy in VALID_VAL_STRATEGIES:
        suffix = f'_val_{strategy}'
        if base.endswith(suffix):
            rest = base[:-len(suffix)]
            break
    else:
        return None

    for setting in sorted(VALID_SETTINGS, key=len, reverse=True):
        prefix = f'{setting}_'
        if rest.startswith(prefix):
            rest2 = rest[len(prefix):]
            break
    else:
        return None

    for target in VALID_TARGETS:
        prefix2 = f'{target}_'
        if rest2.startswith(prefix2):
            model_name = rest2[len(prefix2):]
            return setting, target, model_name, strategy

    return None


def load_all_submissions():
    """
    Walk submissions/ and load all valid metric CSVs into one DataFrame.
    Adds a 'val_strategy' column derived from the filename.

    Returns:
        pd.DataFrame with all submissions combined, or empty DataFrame if none found.
    """
    submissions_dir = os.path.abspath(SUBMISSIONS_DIR)
    if not os.path.isdir(submissions_dir):
        logger.error(f"Submissions directory not found: {submissions_dir}")
        return pd.DataFrame()

    frames = []
    for model_folder in sorted(os.listdir(submissions_dir)):
        folder_path = os.path.join(submissions_dir, model_folder)
        if not os.path.isdir(folder_path):
            continue
        for filename in sorted(os.listdir(folder_path)):
            parsed = parse_submission_filename(filename)
            if parsed is None:
                logger.warning(f"Skipping unrecognised filename: {model_folder}/{filename}")
                continue
            setting, target, model_name, val_strategy = parsed
            filepath = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(filepath)
            except Exception as e:
                logger.warning(f"Could not read {filepath}: {e}")
                continue
            df['val_strategy'] = val_strategy
            frames.append(df)
            logger.info(f"Loaded {model_folder}/{filename} ({len(df)} rows)")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Drop scales not shown on leaderboard
    if 'scale' in combined.columns:
        combined = combined[~combined['scale'].isin(DROP_SCALES)]
        combined['scale'] = combined['scale'].replace({'spatial': 'site-mean'})

    # Keep only the three benchmark settings
    if 'setting' in combined.columns:
        combined = combined[combined['setting'].isin(VALID_SETTINGS)]

    return combined


def build_index_html(targets_with_files):
    """Generate docs/index.html listing all available leaderboard pages."""
    rows = []
    for target in sorted(targets_with_files):
        median_file = f"leaderboard_{target}.html"
        q90_file = f"leaderboard_q90_{target}.html"
        rows.append(f"""
    <section>
      <h2>{target}</h2>
      <ul>
        <li><a href="{median_file}">Median RMSE — {target}</a></li>
        <li><a href="{q90_file}">90th-percentile RMSE — {target}</a></li>
      </ul>
    </section>""")

    sections = "\n".join(rows)
    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FLUXNET ML Leaderboard</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header>
    <h1>FLUXNET ML Leaderboard</h1>
  </header>
  <main>
    <p>
      This leaderboard tracks machine-learning model performance on the FLUXNET
      benchmark across three evaluation settings (temporal split, spatial split,
      temperature-based split) and three carbon/water flux targets (GPP, ET, NEE).
      Metrics are pre-computed by submitters and validated automatically on pull
      request. See the
      <a href="https://github.com/{'{TODO: owner/repo}'}">GitHub repository</a>
      for submission instructions.
    </p>
    {sections}
    <footer>
      <p>
        <a href="https://github.com/{'{TODO: owner/repo}'}">GitHub</a> ·
        Submit your model via pull request
      </p>
    </footer>
  </main>
</body>
</html>"""


def main():
    results = load_all_submissions()
    if results.empty:
        logger.error("No submissions found — nothing to build.")
        sys.exit(1)

    os.makedirs(os.path.abspath(DOCS_DIR), exist_ok=True)

    targets_built = []
    for target in sorted(results['target'].unique()):
        target_df = results[results['target'] == target]

        median_path = os.path.join(DOCS_DIR, f"leaderboard_{target}.html")
        create_html_leaderboard(
            target_df,
            target=target,
            metric='rmse',
            aggfunc='median',
            settings_names=DISPLAY_NAMES,
            filename=median_path,
            wrap_html=True,
            page_title=f"FLUXNET Leaderboard — {target} (median RMSE)",
            page_heading=f"{target} — median RMSE",
        )

        q90_path = os.path.join(DOCS_DIR, f"leaderboard_q90_{target}.html")
        create_html_leaderboard(
            target_df,
            target=target,
            metric='rmse',
            aggfunc=lambda x: x.quantile(0.9),
            settings_names=DISPLAY_NAMES,
            filename=q90_path,
            wrap_html=True,
            page_title=f"FLUXNET Leaderboard — {target} (90th-pct RMSE)",
            page_heading=f"{target} — 90th-percentile RMSE",
        )

        targets_built.append(target)
        logger.info(f"Built leaderboards for {target}")

    index_path = os.path.join(DOCS_DIR, 'index.html')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(build_index_html(targets_built))
    logger.info(f"Built index: {index_path}")

    print(f"\nLeaderboard built for targets: {', '.join(targets_built)}")
    print(f"Output: {os.path.abspath(DOCS_DIR)}/")


if __name__ == '__main__':
    main()
