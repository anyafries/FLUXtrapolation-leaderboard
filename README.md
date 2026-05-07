# FLUXNET ML Leaderboard

Track and compare ML model performance on the [FLUXNET](https://fluxnet.org/) benchmark.
**Live leaderboard → {TODO: fill in GitHub Pages URL, e.g. https://your-org.github.io/leaderboard}**

---

## How to submit

No local clone required — everything is done through GitHub's web UI.

1. **Fork this repo** — click "Fork" in the top-right corner of the GitHub page.
2. **Navigate to `submissions/`** in your fork.
3. **Create a folder** named after your model (lowercase, alphanumeric, hyphens/underscores — e.g. `my-model`).
4. **Upload your 9 CSV files** — click "Add file" → "Upload files" and drag them in.
   Each file must follow the naming and format described below.
5. **Commit** to a branch in your fork, then **open a pull request** back to this repo's `main`.
6. A validation bot will check your files and post a pass/fail comment.
7. Once the checks pass, a maintainer will merge your PR and the leaderboard updates automatically.

---

## File naming and format

A submission is exactly **9 metric CSV files**: every combination of **3 settings × 3 targets**, for one `val_strategy`.

### Folder layout

```
submissions/{model_name}/{setting}_{target}_{model_name}_val_{val_strategy}.csv
```

### Naming rules

| Field | Valid values |
|---|---|
| `model_name` | lowercase alphanumeric + hyphens/underscores (must match the parent folder) |
| `setting` | `time-split`, `spatial-easy40`, `TA40` |
| `target` | `GPP`, `ET`, `NEE` |
| `val_strategy` | `mean`, `max`, `discrepancy` |

### CSV columns (exact order required)

```
target,setting,model,scale,env,n_samples,mse,rmse,mae,nse,r2_score,bias,relative_mae,relative_bias
```

- The `model` column must contain exactly your `model_name` in every row.
- The `setting` and `target` columns must match the filename.
- `n_samples`, `mse`, `rmse`, and `mae` must not have NaN values.

### Updating an existing submission

You may later add a second set of 9 files for a different `val_strategy` (e.g., first submit `mean`, then `max`).
To update files already on `main`, open a new PR modifying any subset of your 9 files for that `val_strategy`.

---

## What gets evaluated

The leaderboard displays **RMSE** (Root Mean Squared Error) aggregated by:

- **Temporal scale** — hourly, weekly, seasonal, inter-annual variability (IAV), anomalies
- **Evaluation setting** — temporal split, spatial split, temperature-based split
- **Target variable** — GPP (Gross Primary Productivity), ET (Evapotranspiration), NEE (Net Ecosystem Exchange)

A **Skill Score** column summarises performance relative to the `lr` baseline across all settings and scales (higher = better).

> **Note:** The MVP trusts submitters' metric numbers. The validation bot checks column names, value types, and file completeness, but does not recompute metrics from raw predictions.

---

## Local development (for maintainers)

```bash
# Install dependencies
pip install -r requirements.txt

# Regenerate docs/ from submissions/
python scripts/build_leaderboard.py

# Open docs/index.html in a browser to preview
open docs/index.html          # macOS
xdg-open docs/index.html      # Linux

# Validate a set of CSV files manually
python scripts/validate_submission.py submissions/lr/*.csv
```

---

## Maintainer setup

### Enable GitHub Pages

Do this **once** after creating the repo:

1. Go to **Settings → Pages**.
2. Under **Source**, select **"Deploy from a branch"**.
3. Branch: `main`, Folder: `/docs`.
4. Click **Save**.

The site will be live at `https://{your-org}.github.io/{repo-name}` within a minute.

### Fork PR comment permissions

The `validate-pr` workflow uses `pull_request` (not `pull_request_target`) for security.
This means it runs with a read-only token for fork PRs, so the bot comment may not appear
for external contributors — the result is still visible in the Actions log.
If you want comments on fork PRs, add a separate `pull_request_target` workflow that reads
the saved report artifact and posts it; this is out of scope for the MVP.
