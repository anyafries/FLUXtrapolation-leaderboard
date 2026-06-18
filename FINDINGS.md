# Phase 0 — Orientation findings

Read-only pass over the leaderboard repo (and the sibling `../code` tree) to answer the
Phase 0 questions before building the upload service. File/line refs are clickable.

---

## TL;DR — the headline

**The repo as it stands today is a *metrics-CSV* leaderboard, not a *raw-predictions* one.**
This is the single most important thing to internalise before Phase 1, because the v5 plan
assumes raw predictions and a recompute step that **do not yet exist here**.

- A "submission" today = **9 pre-computed metric CSVs** (3 settings × 3 targets), one row per
  `(scale, env)`, columns `target,setting,model,scale,env,n_samples,mse,rmse,mae,nse,r2_score,bias,relative_mae,relative_bias`.
  See an example: [submissions/lr/spatial-easy40_GPP_lr_val_mean.csv](submissions/lr/spatial-easy40_GPP_lr_val_mean.csv) (~57 KB, 320 data rows).
- [scripts/build_leaderboard.py](scripts/build_leaderboard.py) reads those metric CSVs **directly** and renders the page. It
  **deliberately does not recompute** — its own docstring says so ([build_leaderboard.py:8-12](scripts/build_leaderboard.py#L8-L12)),
  and the README confirms "the MVP trusts submitters' metric numbers… does not recompute metrics
  from raw predictions" ([README.md:69](README.md#L69)).
- The vendored [eval.py](eval.py) / [utils/eval_utils.py](utils/eval_utils.py) are present but **currently un-runnable** in this
  repo (see "Blocking gaps" below).
- Trust model today is **human-gated merge** ([README.md:19](README.md#L19)), the opposite of the v5 auto-merge goal.

So the v5 plan ("recompute from raw predictions on the VM; eval.py already reads CSV") is only
half-true: the *metric* computation reads CSV fine, but the *raw-prediction loading* and *temporal
aggregation* it depends on are not vendored, and the submission format itself must change from
metrics → raw predictions. None of this is a blocker for the plan; it just relocates work into
Phase 1 (vendor the missing pieces + redefine the submission format) and Phase 4 (the truth-join adapter).

---

## Q1 — Does the metric path read `y_true` from the prediction file, or join it from a separate source?

**It reads `y_true` directly from the predictions DataFrame. There is no join from any external source.**

Call chain:
- [eval.py:36-39](eval.py#L36-L39): `load_predictions(...)` → `compute_and_save_metrics(predictions_df, ...)`.
- [eval_utils.py:119-124](utils/eval_utils.py#L119-L124) `compute_and_save_metrics` → [eval_utils.py:147](utils/eval_utils.py#L147) `compute_metrics`.
- The actual metric read is [eval_utils.py:176](utils/eval_utils.py#L176): `y_true, y_pred = group['y_true'].values, group['y_pred'].values`.

So `predictions_df` must already carry the truth column. Required input columns (inferred from
the code, since `load_predictions` itself is in the missing `dataloader.py`):
- `y_true`, `y_pred` — the metric inputs ([eval_utils.py:176](utils/eval_utils.py#L176)).
- `env` — the grouping key for metrics ([eval_utils.py:175](utils/eval_utils.py#L175)).
- `site_id` — overrides `env` for the multi-year scales `{seasonal, iav, anom}` ([eval_utils.py:165](utils/eval_utils.py#L165)).
- `time` — needed by the temporal `AGGREGATIONS` resampling (in the missing `utils/aggregation.py`).

This matches the confirmed schema in the plan (`y_true, y_pred, env, site_id, time`; join key `(site_id, time)`).

**Adapter for the v5 "score against lr truth" requirement is trivial in shape:** after loading the
submitter's raw file, `LEFT JOIN` the lr-derived truth table on `(site_id, time)`, **overwrite the
`y_true` column with lr's values, drop the submitted `y_true`**, then call `compute_metrics`
unchanged. Because `y_true` is just a column the metric functions read positionally, no change to
eval internals is required.

---

## Q2 — How does eval treat missing / extra `(site_id, time)` rows?

**Eval is fully permissive — it computes over whatever rows are present and enforces no completeness.**

- `compute_metrics` groups the aggregated frame by `env` and computes metrics per group over
  exactly the rows it finds ([eval_utils.py:174-190](utils/eval_utils.py#L174-L190)). `n_samples` is just `len(group)`
  ([eval_utils.py:180](utils/eval_utils.py#L180)) — a dropped/cherry-picked subset simply yields a smaller
  `n_samples` and a (potentially misleadingly better) score, with no error.
- NaNs in `y_true`/`y_pred` are masked out by the metric functions (`np.nanmean`, `np.isfinite`
  masks — e.g. [eval_utils.py:31](utils/eval_utils.py#L31), [eval_utils.py:46](utils/eval_utils.py#L46), [eval_utils.py:80-81](utils/eval_utils.py#L80-L81)). Extra rows are silently included.

**Consequence (confirms the plan):** completeness and anti-cherry-picking **must be enforced in the
validation module, not by eval** — eval will never reject an incomplete submission. The plan's
"completeness check is always on, using the lr baseline as the index reference" is the right call.

**Caveat / open item:** *how a partial group rolls up* (e.g. does a week missing half its hours get
dropped, NaN'd, or averaged over what's present?) lives in the **missing** `utils/aggregation.py`
(`AGGREGATIONS`, called at [eval_utils.py:171](utils/eval_utils.py#L171)). I cannot characterise the exact
resampling/NaN-handling per scale until that module is vendored in Phase 1. Flagging this rather
than guessing.

---

## Q3 — Does the lr baseline cover the full required index (so it can be truth + completeness ref)?

**Not as it exists in this repo — and the raw lr predictions needed to build the truth table are absent. This is a blocking input.**

- The lr files in [submissions/lr/](submissions/lr/) are **metric CSVs**, not raw predictions. They contain
  per-`(scale, env)` metrics, **not** a `(site_id, time) → y_true` index. So today's lr submission
  **cannot** serve as the truth/index source for a raw-prediction flow.
- A search for raw prediction files turned up **nothing**: no `*_predictions.csv` anywhere under
  `/Users/anfries/polybox/fluxnet` (outside the venv). `utils.py` expects them at
  `results/models/{setting}_{target}_{model}_val_{strategy}_predictions.csv`
  ([utils.py:38-41](utils/utils.py#L38-L41), [utils.py:75-99](utils/utils.py#L75-L99)), and that `results/` dir does not exist.

What I *can* confirm about index *shape* from the metric files (useful for the completeness ref later):
- Scales present in lr: `anom, daily, hourly, iav, monthly, seasonal, spatial, weekly` (8); the
  leaderboard shows 6 (daily+monthly dropped, `spatial`→`site-mean`).
- Env coverage at hourly scale: **spatial-easy40 = 40 sites**, **time-split = 340 site-years**
  (the `env` granularity differs by setting, consistent with the column counts in the page).

**Action:** Phase 1 needs the **raw lr `*_predictions.csv`** (the canonical y_true + index) sourced
from wherever the benchmark generated them (the `../code` pipeline / its `results/`). Until those
land, the truth table cannot be built and the recompute path cannot be validated end-to-end.

---

## Q4 — Exact metric call, and the renderer's expected input

### Metric call
`compute_metrics(predictions_df, model_name, setting, target, scales=None, metrics=None)`
([eval_utils.py:147](utils/eval_utils.py#L147)), normally reached via
`compute_and_save_metrics(predictions_df, setting, target, model_name, val_strategy)`
([eval_utils.py:119](utils/eval_utils.py#L119)).
- **Input:** long DataFrame with `y_true, y_pred, env, site_id, time` (see Q1).
- **Output:** one row per `(scale, env)` with columns
  `target, setting, model, scale, env, n_samples, mse, rmse, mae, nse, r2_score, bias, relative_mae, relative_bias`
  ([eval_utils.py:104-113](utils/eval_utils.py#L104-L113), [eval_utils.py:206-210](utils/eval_utils.py#L206-L210)). **This output schema is exactly the current
  submission-CSV schema** — i.e. today's submission file *is* the saved output of this call.

### Renderer
`create_html_leaderboard(df, target, metric, aggfunc, settings_names, return_html=True)`
([plots.py:478](utils/plots.py#L478)), driven by `get_pivot_df_with_scores` ([plots.py:319](utils/plots.py#L319)):
- Expects a long DataFrame with `target, setting, model, scale, env, <metric>` (+ optional
  `val_strategy`, which becomes part of the row index — [plots.py:329-330](utils/plots.py#L329-L330)).
- Drops `scale == 'spatial'` ([plots.py:326](utils/plots.py#L326)); pivots to `index=(model[,val_strategy])`,
  `columns=(setting, scale)`, `values=metric`, `aggfunc` ([plots.py:332-337](utils/plots.py#L332-L337)).
- Computes a **skill score vs `baseline_model='lr'`** ([plots.py:354](utils/plots.py#L354), [plots.py:409-450](utils/plots.py#L409-L450)) — so **lr
  must always be present** or the Summary column silently disappears ([plots.py:425-432](utils/plots.py#L425-L432)).
- Scale order: `['hourly','daily','weekly','monthly','seasonal','anom','iav']` ([plots.py:485](utils/plots.py#L485)); setting
  order from `SETTINGS_ORDER` ([plots.py:27](utils/plots.py#L27)); model order via skill score unless overridden.

### Page layout (CONFIRM: keep this)
[build_leaderboard.py](scripts/build_leaderboard.py) produces a **single tabbed [docs/index.html](docs/index.html)** — one tab per target
(`ET`, `GPP`, `NEE`, [build_leaderboard.py:33](scripts/build_leaderboard.py#L33)), each tab showing **two tables: "Median RMSE"
and "90th-percentile RMSE"** ([build_leaderboard.py:253-269](scripts/build_leaderboard.py#L253-L269)). Columns are grouped by setting
(temporal / spatial / temperature) × scale (hourly, weekly, seasonal, anom, iav, site-mean) with a
leading **Summary "Skill score ↑"** column. `build_leaderboard.py` pre-drops `daily`/`monthly`
([build_leaderboard.py:41](scripts/build_leaderboard.py#L41), [build_leaderboard.py:129](scripts/build_leaderboard.py#L129)) and renames `spatial`→`site-mean` ([build_leaderboard.py:130](scripts/build_leaderboard.py#L130)).
**This is the layout to preserve.** The recompute flow must feed this renderer the same long-format
metrics it gets today.

---

## "To confirm during Phase 0" checklist

- [x] **How eval handles missing rows** — permissive, no completeness enforcement (Q2). Exact per-scale
  roll-up behaviour is gated on vendoring `utils/aggregation.py`.
- [ ] **lr baseline covers the full required index** — **cannot confirm; raw lr predictions are absent** (Q3).
  Must source the raw lr `*_predictions.csv` before the truth table / completeness ref can exist.
- [ ] **Archive target** (campus research-storage mount vs Dropbox/Drive) — not determinable from code;
  needs a maintainer decision. No archive pointer plumbing exists yet.
- [ ] **CSV → Parquet later** — `requirements.txt` has no `pyarrow`/`fastparquet` ([requirements.txt:1-5](requirements.txt#L1-L5)); CSV-only for now, as planned.

---

## Blocking gaps & required changes for the v5 plan (feeds Phase 1/2)

1. **Vendored eval pipeline is import-broken.** `import utils.eval_utils` fails at
   [eval_utils.py:13](utils/eval_utils.py#L13): `from utils.aggregation import AGGREGATIONS` — **`utils/aggregation.py` does not
   exist** in this repo, and neither does **`dataloader.py`** (`eval.py:14` imports `load_predictions`
   from it). `build_leaderboard.py` only dodges this by importing `utils.plots`/`utils.utils`, never
   `eval_utils`. **Phase 1 must vendor `utils/aggregation.py` and `dataloader.py` (or replace the
   loader with the truth-join adapter) and add a clean-install smoke test**, or the VM recompute is dead on arrival.
2. **No raw lr predictions** → no truth table, no completeness reference (Q3). Source them first.
3. **Submission format must change** from metric CSV → raw prediction CSV (`y_true,y_pred,env,site_id,time`,
   ~1.1 GB / 9 files). `validate_submission.py` and `build_leaderboard.py` currently assume the
   metric schema and will need a recompute step inserted between intake and render.
4. **Validation is metrics-shaped and missing two v5 checks.** [scripts/validate_submission.py](scripts/validate_submission.py) checks
   filename/columns/completeness for *metric* CSVs ([validate_submission.py:28-31](scripts/validate_submission.py#L28-L31), [validate_submission.py:224](scripts/validate_submission.py#L224)) but has
   **no `y_true`-vs-lr integrity check and no ownership/`owner`-token concept** (no `metadata.yaml`
   anywhere). Both are net-new for Phase 2.
5. **Trust model is inverted.** Today: human merges ([README.md:19](README.md#L19)); `validate-pr.yml` runs read-only,
   no auto-merge ([.github/workflows/validate-pr.yml](.github/workflows/validate-pr.yml)). v5 wants Action-driven auto-merge on pass.
6. **No intake infra at all** — no Uppy drop box, no R2, no Worker, no `metadata.yaml` schema. Entirely Phase 3.

## Minor issues noted in passing (non-blocking)

- `GITHUB_REPO_URL`/Pages URL are still `{TODO: owner/repo}` placeholders ([build_leaderboard.py:48](scripts/build_leaderboard.py#L48),
  [docs/index.html:16](docs/index.html#L16), [README.md:4](README.md#L4)).
- Existing submissions are **not** clean single-strategy 9-sets: `mlp` and `mmd` each have a stray
  `TA40_NEE_*_val_max.csv` mixed in with `val_mean` files — relevant to how the completeness rule
  groups by `(model, val_strategy)` ([validate_submission.py:237](scripts/validate_submission.py#L237)).
- `find_available_experiments` docstring says it returns 3-tuples but actually returns 4-tuples incl.
  strategy ([utils.py:79-99](utils/utils.py#L79-L99)).
- `requirements.txt` omits `pyyaml` (needed once `metadata.yaml` arrives) and `pyarrow` (future Parquet).
