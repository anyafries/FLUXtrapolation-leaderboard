"""
Build the FLUXNET leaderboard HTML pages from submission CSVs.

Reads all metric CSVs under submissions/{model_name}/ and generates:
  docs/index.html  — single page with flux (ET / GPP / NEE), RMSE aggregation
                     (90th pct / median) and shift filters; one table per
                     flux × aggregation, the shift filter is applied client-side.

Note: This script directly scans submissions/ rather than using eval.py's
load_all_metrics(), because the existing eval.py depends on dataloader.py and
utils/aggregation.py for recomputing metrics from raw predictions. Since
submitters provide pre-computed metric CSVs, no recomputation is needed.
"""

import hashlib
import os
import re
import sys
import pandas as pd
import yaml

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

# Full flux names shown in the table headings, e.g. "Median RMSE - Evapotranspiration (ET)".
TARGET_FULL_NAMES = {
    'ET': 'Evapotranspiration',
    'GPP': 'Gross primary production',
    'NEE': 'Net ecosystem exchange',
}

# ET values are displayed scaled by 100 (see create_html_leaderboard); say so under the heading.
HEADING_NOTES = {'ET': 'values scaled by 100'}

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

GITHUB_REPO_URL = "https://github.com/anyafries/FLUXtrapolation-leaderboard"
PAPER_URL = "https://arxiv.org/abs/2605.19812"
BENCHMARK_REPO_URL = "https://github.com/anyafries/FLUXtrapolation"
SUBMIT_URL = "submit.html"


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


# Short labels for the narrow "Val." column (used when the submitter gives no val_strategy_display).
VAL_ABBREV = {'discrepancy': 'disc'}


def load_display_map():
    """Map (model_id, val_strategy) -> per-row display attributes read from metadata.yaml.

    Returns, for each submission, the submitter-chosen labels plus the provenance/trust
    attributes the leaderboard renders as extra row-heading columns:
      {'model': display_name, 'val': val_strategy_display,
       'institution': institution, 'reviewed': bool, 'is_baseline': bool,
       'code_url': code_url}
    Missing labels fall back to the raw id (abbreviated via VAL_ABBREV for the val strategy);
    a missing institution stays None (renders '-'). When code_url is set the model name is
    rendered as a link to it (opens in a new tab).
    """
    submissions_dir = os.path.abspath(SUBMISSIONS_DIR)
    out = {}
    if not os.path.isdir(submissions_dir):
        return out
    for folder in sorted(os.listdir(submissions_dir)):
        meta_path = os.path.join(submissions_dir, folder, 'metadata.yaml')
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, encoding='utf-8') as f:
                meta = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Could not read {meta_path}: {e}")
            continue
        model_id = meta.get('model_id')
        val_strategy = meta.get('val_strategy')
        if not model_id or not val_strategy:
            continue
        out[(model_id, val_strategy)] = {
            'model': meta.get('display_name') or model_id,
            'val': meta.get('val_strategy_display') or VAL_ABBREV.get(val_strategy, val_strategy),
            'institution': meta.get('institution'),
            'reviewed': bool(meta.get('reviewed', False)),
            'is_baseline': bool(meta.get('is_baseline', False)),
            'code_url': meta.get('code_url') or None,
        }
    return out


def _stylesheet_version():
    """Short content hash of docs/style.css, used as a cache-busting query string on the
    stylesheet link so browsers (and GitHub Pages' CDN) fetch fresh CSS after every change."""
    css_path = os.path.join(DOCS_DIR, 'style.css')
    try:
        with open(css_path, 'rb') as f:
            return hashlib.sha1(f.read()).hexdigest()[:8]
    except OSError:
        return '0'


PAGE_SCRIPT = r"""
  (function () {
    var state = { tab: FIRST_TAB, agg: 'q90', shift: 'all' };
    var AGG_LABELS = { q90: '90th-percentile', median: 'Median' };

    function setPills(selector, key, value) {
      document.querySelectorAll(selector).forEach(function (b) {
        var on = b.dataset[key] === value;
        b.classList.toggle('active', on);
        b.setAttribute('aria-pressed', on);
        if (b.getAttribute('role') === 'tab') b.setAttribute('aria-selected', on);
      });
    }

    function activeTable() {
      var panel = document.querySelector(
        '[role="tabpanel"][data-tab="' + state.tab + '"] .agg-panel[data-agg="' + state.agg + '"]');
      return panel ? panel.querySelector('table') : null;
    }

    function scoreOf(ss) {
      var a = ss.getAttribute('data-ss-' + state.shift);
      return a === null ? NaN : parseFloat(a);
    }
    function fmt(v) { return isNaN(v) ? '-' : v.toFixed(3); }

    // Show only the selected shift's column block, swap the skill score to that shift's score
    // and re-sort the rows by it. All scores are baked into the skill cell's data attributes.
    function applyShift(table) {
      table.querySelectorAll('th.col_heading.level0').forEach(function (th) {
        var span = parseInt(th.getAttribute('colspan') || '1', 10);
        if (span < 2) return;                       // the skill-score column
        var m = th.className.match(/(?:^|\s)col(\d+)(?:\s|$)/);
        if (!m) return;
        var start = parseInt(m[1], 10);
        var show = state.shift === 'all' || th.textContent.trim() === state.shift;
        for (var i = start; i < start + span; i++) {
          table.querySelectorAll('.col' + i).forEach(function (c) {
            c.classList.toggle('col-hidden', !show);
          });
        }
      });
      var tbody = table.tBodies[0];
      var rows = Array.prototype.slice.call(tbody.rows);
      rows.forEach(function (tr) {
        var ss = tr.querySelector('.ss');
        var v = ss ? scoreOf(ss) : NaN;
        tr.dataset.score = v;
        if (ss) ss.textContent = ss.hasAttribute('data-baseline') ? '-' : fmt(v);
      });
      rows.sort(function (a, b) {
        var x = parseFloat(a.dataset.score), y = parseFloat(b.dataset.score);
        if (isNaN(x) && isNaN(y)) return 0;
        if (isNaN(x)) return 1;
        if (isNaN(y)) return -1;
        return y - x;
      });
      rows.forEach(function (tr) { tbody.appendChild(tr); });
    }

    // Top-3 reviewed models (baselines included) for the current flux / aggregation / shift.
    function renderCards(table) {
      var wrap = document.getElementById('leader-cards');
      var items = [];
      table.querySelectorAll('tbody .ss').forEach(function (ss) {
        if (!ss.hasAttribute('data-reviewed')) return;
        var v = scoreOf(ss);
        if (isNaN(v)) return;
        items.push({ name: ss.getAttribute('data-model') || '', score: v });
      });
      items.sort(function (a, b) { return b.score - a.score; });
      items = items.slice(0, 3);
      wrap.innerHTML = '';
      items.forEach(function (it, i) {
        var card = document.createElement('div');
        card.className = 'card' + (i === 0 ? ' top' : '');
        var pct = Math.max(0, Math.min(1, it.score)) * 100;
        card.innerHTML =
          '<div class="card-head"><span class="rank">#' + (i + 1) + '</span>' +
          '<span class="name"></span><span class="score">' + it.score.toFixed(3) + '</span></div>' +
          '<div class="bar"><span style="width:' + pct.toFixed(1) + '%"></span></div>';
        card.querySelector('.name').textContent = it.name;
        wrap.appendChild(card);
      });
      document.getElementById('leaders-empty').hidden = items.length > 0;
    }

    function updateHeadings() {
      var agg = AGG_LABELS[state.agg];
      var full = FULL_NAMES[state.tab] || state.tab;
      document.getElementById('section-title').textContent =
        agg + ' RMSE \u2014 ' + full + ' (' + state.tab + ')';
      document.getElementById('results-desc').textContent =
        agg + ' RMSE for all models, across all timescales, ' +
        (state.shift === 'all' ? 'and all extrapolation scenarios.'
                               : 'for the ' + state.shift + ' scenario.');
      var note = HEADING_NOTES[state.tab] || '';
      document.getElementById('sorted-by').textContent =
        'sorted by skill score' + (note ? ', RMSE ' + note : '');
    }

    function render() {
      setPills('[role="tab"]', 'target', state.tab);
      setPills('.agg-btn', 'agg', state.agg);
      setPills('.shift-btn', 'shift', state.shift);
      document.querySelectorAll('[role="tabpanel"]').forEach(function (p) {
        p.hidden = p.dataset.tab !== state.tab;
      });
      document.querySelectorAll('.agg-panel').forEach(function (p) {
        p.hidden = p.dataset.agg !== state.agg;
      });
      var table = activeTable();
      if (table) {
        applyShift(table);
        renderCards(table);
        table.parentNode.classList.toggle('fill', state.shift === 'all');
        table.classList.toggle('single', state.shift !== 'all');
      }
      updateHeadings();
      history.replaceState(null, '', '#' + state.tab);
    }

    document.querySelectorAll('[role="tab"]').forEach(function (b) {
      b.addEventListener('click', function () { state.tab = b.dataset.target; render(); });
    });
    document.querySelectorAll('.agg-btn').forEach(function (b) {
      b.addEventListener('click', function () { state.agg = b.dataset.agg; render(); });
    });
    document.querySelectorAll('.shift-btn').forEach(function (b) {
      b.addEventListener('click', function () { state.shift = b.dataset.shift; render(); });
    });

    var hash = location.hash.slice(1);
    if (VALID_TABS.indexOf(hash) !== -1) state.tab = hash;
    render();

    // Disclosure buttons ("What is the skill score?", "How to read the table"): each toggles
    // the block named by aria-controls; the label stays put and only the arrow flips.
    document.querySelectorAll('.howto-btn').forEach(function (btn) {
      var block = document.getElementById(btn.getAttribute('aria-controls'));
      if (!block) return;
      btn.addEventListener('click', function () {
        var open = block.hidden;
        block.hidden = !open;
        btn.setAttribute('aria-expanded', open);
        btn.textContent = btn.dataset.label + ' ' + (open ? '\u2191' : '\u2193');
      });
    });

    // Column hover highlight (rows are handled in CSS via tr:hover).
    document.querySelectorAll('table').forEach(function (table) {
      function clearCols() {
        table.querySelectorAll('.hl-col').forEach(function (c) { c.classList.remove('hl-col'); });
      }
      table.addEventListener('mouseover', function (e) {
        var cell = e.target.closest('td, th');
        if (!cell) return;
        clearCols();
        var m = cell.className.match(/(?:^|\s)(col\d+)(?:\s|$)/);
        if (m) {
          table.querySelectorAll('.' + m[1]).forEach(function (c) {
            if (!c.classList.contains('level0')) c.classList.add('hl-col');
          });
        }
      });
      table.addEventListener('mouseleave', clearCols);
    });
  })();
"""


def site_header(active):
    """Shared top bar: wordmark left, nav right. The four nav items keep the same order and
    spacing on every page; the current page (`active` = 'leaderboard' or 'submit') is the
    underlined plain link and the other page's entry is the orange button. (docs/submit.html
    is hand-written — keep its header markup in sync with this.)"""
    lb = 'active' if active == 'leaderboard' else 'btn-primary'
    sb = 'active' if active == 'submit' else 'btn-primary'
    return f"""\
  <header class="site-header">
    <a class="brand" href="index.html"><span class="flux">FLUX</span>trapolation <span class="brand-tag">Benchmark</span></a>
    <nav class="site-nav" aria-label="Site">
      <a class="{lb}" href="index.html">Leaderboard</a>
      <a href="{PAPER_URL}" target="_blank" rel="noopener">Paper <span class="ext" aria-hidden="true">↗</span></a>
      <a href="{BENCHMARK_REPO_URL}" target="_blank" rel="noopener">GitHub <span class="ext" aria-hidden="true">↗</span></a>
      <a class="{sb}" href="{SUBMIT_URL}">Submit a result</a>
    </nav>
  </header>"""


def stamp_submit_stylesheet(css_version):
    """docs/submit.html is hand-written; give its stylesheet link the same cache-busting
    version as index.html so both pages always load the current CSS."""
    path = os.path.join(os.path.abspath(DOCS_DIR), 'submit.html')
    if not os.path.isfile(path):
        return
    with open(path, encoding='utf-8') as f:
        html = f.read()
    stamped = re.sub(r'href="style\.css(?:\?v=[0-9a-f]*)?"', f'href="style.css?v={css_version}"', html)
    if stamped != html:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(stamped)
        logger.info(f"Stamped stylesheet version into {path}")


def build_tabbed_index(tab_panels):
    """
    Build a single tabbed index.html.

    Args:
        tab_panels: dict mapping target -> {'median': table_html, 'q90': table_html}
    """
    present = [t for t in TAB_ORDER if t in tab_panels]
    first_tab = present[0] if present else TAB_ORDER[0]
    tabs_js = '[' + ', '.join(f'"{t}"' for t in present) + ']'
    full_names_js = '{' + ', '.join(f'"{t}": "{TARGET_FULL_NAMES.get(t, t)}"' for t in present) + '}'
    notes_js = '{' + ', '.join(f'"{t}": "{n}"' for t, n in HEADING_NOTES.items() if t in present) + '}'

    buttons = []
    panels = []
    for target in present:
        is_first = target == first_tab
        aria = "true" if is_first else "false"
        cls = ' class="pill active"' if is_first else ' class="pill"'
        hidden_attr = '' if is_first else ' hidden'
        buttons.append(
            f'          <button role="tab" data-target="{target}" '
            f'aria-selected="{aria}"{cls}>{target}</button>'
        )
        median_html = tab_panels[target]['median']
        q90_html = tab_panels[target]['q90']
        panels.append(
            f'      <div role="tabpanel" data-tab="{target}"{hidden_attr}>\n'
            f'        <div class="agg-panel" data-agg="q90">\n'
            f'          <div class="table-scroll">{q90_html}</div>\n'
            f'        </div>\n'
            f'        <div class="agg-panel" data-agg="median" hidden>\n'
            f'          <div class="table-scroll">{median_html}</div>\n'
            f'        </div>\n'
            f'      </div>'
        )

    buttons_html = '\n'.join(buttons)
    panels_html = '\n'.join(panels)
    css_version = _stylesheet_version()
    shift_buttons = '\n'.join(
        f'          <button class="pill shift-btn{" active" if key == "all" else ""}" '
        f'data-shift="{key}">{label}</button>'
        for key, label in [('all', 'all three')] + [(v, v) for v in DISPLAY_NAMES.values()]
    )

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FLUXtrapolation Benchmark</title>
  <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>%E2%A4%B4</text></svg>">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="style.css?v={css_version}">
</head>
<body>
{site_header('leaderboard')}
  <main class="page">
    <section class="intro">
      <div class="intro-main">
        <div class="hero">
          <h1><span class="flux">FLUX</span>trapolation leaderboard</h1>
          <p>This leaderboard tracks machine-learning model performance on the FLUXtrapolation
            benchmark for extrapolating ecosystem fluxes. Each model is scored on prediction
            error of three fluxes (ET, GPP, NEE) in three extrapolation scenarios (temporal,
            spatial, temperature) for six timescales.</p>
        </div>
        <div class="controls">
      <div class="ctl-group">
        <span class="ctl-label" id="flux-label">Flux</span>
        <div class="pills" role="tablist" aria-labelledby="flux-label">
{buttons_html}
        </div>
      </div>
      <div class="ctl-group">
        <span class="ctl-label" id="agg-label">RMSE over <br>sites/site-years</span>
        <div class="pills" role="group" aria-labelledby="agg-label">
          <button class="pill agg-btn active" data-agg="q90">90th percentile</button>
          <button class="pill agg-btn" data-agg="median">median</button>
        </div>
      </div>
      <div class="ctl-group">
        <span class="ctl-label" id="shift-label">Extrapolation <br>scenario</span>
        <div class="pills" role="group" aria-labelledby="shift-label">
{shift_buttons}
        </div>
      </div>
        </div>
      </div>
      <img class="hero-img" src="extrapolation.png" width="578" height="515"
           alt="World map of flux tower sites with example ET time series; one held-out series is marked with a question mark">
    </section>

    <h2 class="section-title" id="section-title"></h2>

    <section class="leaders">
      <div class="eyebrow">Skill score leaders</div>
      <div class="results-head">
        <div>
          <p class="section-desc">Highest skill score against the linear-regression baseline, for the filters above (higher is better).</p>
          <p class="section-desc">Only reviewed models shown here.</p>
        </div>
        <button class="howto-btn" type="button" aria-expanded="false" aria-controls="skill-info" data-label="What is the skill score?">What is the skill score? ↓</button>
      </div>
      <div class="howto howto-single" id="skill-info" hidden>
        <div class="howto-col">
          <div class="eyebrow">Skill score</div>
          <p>The skill score compares a model's error with the linear-regression baseline
            <code>lr</code>. For every cell in the table it is
            <code>1 − RMSE(model) / RMSE(lr)</code>, and the score shown is the average of
            these over all timescales and all extrapolation scenarios currently selected.
            <strong>0</strong> means the model does as well as <code>lr</code>,
            <strong>1</strong> would mean zero error, and a negative score means the model is
            worse than <code>lr</code>. Higher is better.</p>
        </div>
      </div>
      <div class="cards" id="leader-cards"></div>
      <p class="section-desc muted" id="leaders-empty" hidden>No reviewed models yet for this selection.</p>
    </section>

    <section class="results">
      <div class="eyebrow">Full results</div>
      <div class="results-head">
        <div>
          <p class="section-desc" id="results-desc"></p>
          <p class="sorted-by" id="sorted-by"></p>
        </div>
        <button class="howto-btn" type="button" aria-expanded="false" aria-controls="howto" data-label="How to read the table">How to read the table ↓</button>
      </div>

      <div class="howto" id="howto" hidden>
        <div class="howto-col">
          <div class="eyebrow">Reading a cell</div>
          <p>Each cell is the aggregated RMSE (median / 90th quantile) of the predicted
            site / site-year flux at that specific timescale (lower is better). Shading is
            relative to the best model in that column: full tint at the best value, none at
            1.2× the best.</p>
          <div class="tint-scale"><span>1.0×</span><span class="tint-bar"></span><span>1.2×</span></div>
        </div>
        <div class="howto-col">
          <div class="eyebrow">Skill score</div>
          <p>One number per model, relative to the linear-regression baseline: <strong>0</strong>
            means on par with <code>lr</code>, <strong>1</strong> means zero error, negative
            means worse than <code>lr</code>.</p>
        </div>
        <div class="howto-col">
          <div class="eyebrow">Timescales</div>
          <dl class="glossary">
            <dt>hourly</dt><dd>raw hourly error</dd>
            <dt>weekly</dt><dd>weekly means</dd>
            <dt>seasonal</dt><dd>mean seasonal cycle</dd>
            <dt>anom</dt><dd>anomalies from the seasonal cycle</dd>
            <dt>iav</dt><dd>interannual variability</dd>
            <dt>site-mean</dt><dd>long-term site mean</dd>
          </dl>
        </div>
      </div>

{panels_html}

      <p class="review-note">
        A <span class="rev-tick">✓</span> after a model name marks a method whose code we have
        checked. For top-performing methods from other institutions or contributors, we manually
        check the submitted code to ensure, for example, that there is no test-set leakage. If we
        find issues, we (temporarily) remove the submission and contact the author.
      </p>
    </section>
  </main>
  <footer class="site-footer">
    <p>For any issues, contact anya[dot]fries[at]stat[dot]math[dot]ethz[dot]ch</p>
    <p>© Copyright 2026 Anya Fries. Hosted by GitHub Pages.</p>
  </footer>
  <script>
    var VALID_TABS = {tabs_js};
    var FIRST_TAB = '{first_tab}';
    var FULL_NAMES = {full_names_js};
    var HEADING_NOTES = {notes_js};
{PAGE_SCRIPT}
  </script>
</body>
</html>"""


def main():
    results = load_all_submissions()
    if results.empty:
        logger.error("No submissions found — nothing to build.")
        sys.exit(1)
    display_map = load_display_map()

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
            index_display=display_map,
            return_html=True,
            inline_styles=False,
        )

        q90_html = create_html_leaderboard(
            target_df,
            target=target,
            metric='rmse',
            aggfunc=lambda x: x.quantile(0.9),
            settings_names=DISPLAY_NAMES,
            index_display=display_map,
            return_html=True,
            inline_styles=False,
        )

        tab_panels[target] = {'median': median_html, 'q90': q90_html}
        logger.info(f"Built leaderboard tables for {target}")

    index_path = os.path.join(docs_dir, 'index.html')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(build_tabbed_index(tab_panels))
    logger.info(f"Built index: {index_path}")
    stamp_submit_stylesheet(_stylesheet_version())

    built = [t for t in TAB_ORDER if t in tab_panels]
    print(f"\nLeaderboard built for targets: {', '.join(built)}")
    print(f"Output: {docs_dir}/")


if __name__ == '__main__':
    main()
