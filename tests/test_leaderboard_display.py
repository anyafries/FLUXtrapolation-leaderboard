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
