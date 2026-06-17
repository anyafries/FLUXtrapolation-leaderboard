"""
Build the FLUXNET leaderboard HTML pages from submission CSVs.

Reads all metric CSVs under submissions/{model_name}/ and generates:
  docs/index.html  — single tabbed page (ET / GPP / NEE), each tab showing
                     median RMSE and 90th-percentile RMSE tables.

Note: This script directly scans submissions/ rather than using eval.py's
load_all_metrics(), because the existing eval.py depends on dataloader.py and
utils/aggregation.py for recomputing metrics from raw predictions. Since
submitters provide pre-computed metric CSVs, no recomputation is needed.
"""

import os
import sys
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

TAB_ORDER = ['ET', 'GPP', 'NEE']

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

GITHUB_REPO_URL = "https://github.com/{TODO: owner/repo}"  # replace before publishing


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
            # metadata.yaml (and any non-CSV) lives beside the metric CSVs; not a metrics file.
            if filename == 'metadata.yaml' or not filename.endswith('.csv'):
                continue
            parsed = parse_submission_filename(filename)
            if parsed is None:
                logger.warning(f"Skipping unrecognised filename: {model_folder}/{filename}")
                continue
            val_strategy = parsed[3]
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


def build_tabbed_index(tab_panels):
    """
    Build a single tabbed index.html.

    Args:
        tab_panels: dict mapping target -> {'median': table_html, 'q90': table_html}
    """
    present = [t for t in TAB_ORDER if t in tab_panels]
    first_tab = present[0] if present else TAB_ORDER[0]
    tabs_js = '[' + ', '.join(f'"{t}"' for t in present) + ']'

    buttons = []
    panels = []
    for target in present:
        is_first = target == first_tab
        aria = "true" if is_first else "false"
        cls = ' class="active"' if is_first else ''
        hidden_attr = '' if is_first else ' hidden'
        buttons.append(
            f'      <button role="tab" data-target="{target}" '
            f'aria-selected="{aria}"{cls}>{target}</button>'
        )
        median_html = tab_panels[target]['median']
        q90_html = tab_panels[target]['q90']
        panels.append(
            f'    <div role="tabpanel" data-tab="{target}"{hidden_attr}>\n'
            f'      <h2>Median RMSE</h2>\n'
            f'      {median_html}\n'
            f'      <h2>90th-percentile RMSE</h2>\n'
            f'      {q90_html}\n'
            f'    </div>'
        )

    buttons_html = '\n'.join(buttons)
    panels_html = '\n'.join(panels)

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FLUXtrapolation Leaderboard</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header>
    <h1>FLUXtrapolation Leaderboard</h1>
    <p>
      This leaderboard tracks machine-learning model performance on the FLUXtrapolation
      benchmark across three evaluation settings (temporal split, spatial split,
      temperature-based split) and three carbon/water flux targets (GPP, ET, NEE).
      See the <a href="{GITHUB_REPO_URL}">GitHub repository</a>
      for submission instructions.
    </p>
  </header>
  <main>
    <div role="tablist" class="tab-bar">
{buttons_html}
    </div>
{panels_html}
    <footer>
      <p>
        <a href="{GITHUB_REPO_URL}">GitHub</a> ·
        Submit your model via pull request
      </p>
    </footer>
  </main>
  <script>
    var VALID_TABS = {tabs_js};
    function activateTab(t) {{
      document.querySelectorAll('[role="tab"]').forEach(function(b) {{
        var on = b.dataset.target === t;
        b.setAttribute('aria-selected', on);
        b.classList.toggle('active', on);
      }});
      document.querySelectorAll('[role="tabpanel"]').forEach(function(p) {{
        p.hidden = p.dataset.tab !== t;
      }});
      history.replaceState(null, '', '#' + t);
    }}
    document.querySelectorAll('[role="tab"]').forEach(function(b) {{
      b.addEventListener('click', function() {{ activateTab(b.dataset.target); }});
    }});
    var hash = location.hash.slice(1);
    activateTab(VALID_TABS.indexOf(hash) !== -1 ? hash : '{first_tab}');
  </script>
</body>
</html>"""


def main():
    results = load_all_submissions()
    if results.empty:
        logger.error("No submissions found — nothing to build.")
        sys.exit(1)

    docs_dir = os.path.abspath(DOCS_DIR)
    os.makedirs(docs_dir, exist_ok=True)

    # Remove stale per-target files from the old multi-file layout
    for target in VALID_TARGETS:
        for old_name in [f'leaderboard_{target}.html', f'leaderboard_q90_{target}.html']:
            old_path = os.path.join(docs_dir, old_name)
            if os.path.exists(old_path):
                os.remove(old_path)
                logger.info(f"Removed old file: {old_path}")

    tab_panels = {}
    for target in TAB_ORDER:
        if target not in results['target'].unique():
            continue
        target_df = results[results['target'] == target]

        median_html = create_html_leaderboard(
            target_df,
            target=target,
            metric='rmse',
            aggfunc='median',
            settings_names=DISPLAY_NAMES,
            return_html=True,
        )

        q90_html = create_html_leaderboard(
            target_df,
            target=target,
            metric='rmse',
            aggfunc=lambda x: x.quantile(0.9),
            settings_names=DISPLAY_NAMES,
            return_html=True,
        )

        tab_panels[target] = {'median': median_html, 'q90': q90_html}
        logger.info(f"Built leaderboard tables for {target}")

    index_path = os.path.join(docs_dir, 'index.html')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(build_tabbed_index(tab_panels))
    logger.info(f"Built index: {index_path}")

    built = [t for t in TAB_ORDER if t in tab_panels]
    print(f"\nLeaderboard built for targets: {', '.join(built)}")
    print(f"Output: {docs_dir}/")


if __name__ == '__main__':
    main()
