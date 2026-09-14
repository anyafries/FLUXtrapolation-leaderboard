"""The leaderboard relabels rows with submitter-chosen display names, while keeping the
skill-score baseline and ordering keyed on the real model_id / val_strategy."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.plots import create_html_leaderboard


def _toy_df():
    """Minimal (model, val_strategy, setting, scale, rmse) frame with an `lr` baseline."""
    rows = []
    for model, rmse in [("lr", 1.0), ("foo", 0.5)]:
        for setting in ["time-split", "spatial-easy40", "TA40"]:
            for scale in ["hourly", "weekly"]:
                rows.append({
                    "target": "ET", "model": model, "val_strategy": "mean",
                    "setting": setting, "scale": scale, "env": "x", "rmse": rmse,
                })
    return pd.DataFrame(rows)


def test_index_display_relabels_rows():
    html = create_html_leaderboard(
        _toy_df(), target="ET", metric="rmse",
        index_display={("foo", "mean"): {"model": "Foo Net", "val": "CV-mean"}},
        return_html=True,
    )
    assert "Foo Net" in html, "custom model display name should appear"
    assert "CV-mean" in html, "custom val_strategy display name should appear"
    # lr has no override -> shows its raw id; baseline still resolved (skill scores rendered).
    assert "lr" in html
    assert "Skill score" in html


def test_no_index_display_keeps_raw_ids():
    html = create_html_leaderboard(_toy_df(), target="ET", metric="rmse", return_html=True)
    assert "foo" in html and "lr" in html
    assert "Foo Net" not in html


def test_institution_column_and_reviewed_tick():
    """Institution row-heading column renders from index_display; a reviewed submission gets a
    ✓ after its model name, outside the code link. The ✓ depends solely on `reviewed`
    (is_baseline does NOT tick a row); an unreviewed row gets none. The skill cell carries the
    reviewed flag + plain model name + per-shift scores for the page's Shift filter / cards."""
    html = create_html_leaderboard(
        _toy_df(), target="ET", metric="rmse",
        index_display={
            ("foo", "mean"): {"model": "Foo Net", "val": "mean", "institution": "ETH Zürich",
                              "reviewed": True, "code_url": "https://example.org/foo"},
            ("lr", "mean"): {"model": "lr", "val": "mean",
                             "institution": None, "reviewed": False, "is_baseline": True},
        },
        return_html=True,
    )
    assert "Institution" in html
    assert "Reviewed" not in html            # no separate column any more
    assert "ETH Zürich" in html
    # foo is reviewed -> exactly one tick, after the closing </a> of the code link.
    assert html.count("✓") == 1
    assert '>Foo Net</a> <span class="rev-tick"' in html
    # Skill cell data attributes: reviewed flag only on foo; per-shift scores on both rows.
    assert html.count('data-reviewed="1"') == 1
    assert 'data-model="Foo Net"' in html and 'data-model="lr"' in html
    for shift in ("time-split", "spatial-easy40", "TA40"):
        assert html.count(f'data-ss-{shift}="') == 2
    assert html.count('data-baseline="1"') == 1


def test_missing_institution_renders_dash():
    """A row whose index_display lacks an institution renders '-' (not a crash)."""
    html = create_html_leaderboard(
        _toy_df(), target="ET", metric="rmse",
        index_display={("foo", "mean"): {"model": "Foo Net", "val": "mean"}},
        return_html=True,
    )
    assert "Institution" in html
    # foo has no institution key -> '-'; no tick anywhere (nothing reviewed).
    assert "✓" not in html
