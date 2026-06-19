# FLUXtrapolation Leaderboard

ML model performance on the [FLUXNET](https://fluxnet.org/) **FLUXtrapolation** benchmark
(temporal / spatial / temperature splits × GPP / ET / NEE).

**Live:** https://anyafries.github.io/FLUXtrapolation-leaderboard/#ET

---

## Run locally

```bash
# 1. Environment (first time only)
python -m venv venv_leaderboard
source venv_leaderboard/bin/activate
pip install -r requirements.txt          # includes jinja2, needed by the pandas Styler

# 2. Build docs/index.html from submissions/
python scripts/build_leaderboard.py

# 3. Preview — serve docs/ so relative links + the favicon resolve like the deployed site
python -m http.server 8000 --directory docs
#   → http://localhost:8000/   (submit page at /submit.html)
```

- **CSS-only tweaks** to [docs/style.css](docs/style.css) just need a browser refresh — no rebuild.
- **Markup/data changes** (header, tabs, new submissions) need `build_leaderboard.py` re-run.
- Run the test suite with `pytest` (covers intake, validation, scoring, truth).

---

## Repo structure

```
docs/                     # the static GitHub Pages site (deployed from /docs on main)
  index.html              #   GENERATED leaderboard — do not edit by hand; rebuild instead
  submit.html             #   upload UI (Uppy drop box → Cloudflare Worker)
  style.css               #   hand-edited stylesheet (shared by both pages)

scripts/                  # maintainer / CI entrypoints
  build_leaderboard.py    #   submissions/*/*.csv → docs/index.html
  score_submissions.py    #   score every `pending` submission (CI: score-and-publish)
  build_truth_table.py    #   lr raw predictions → reference/truth_table.parquet (+ manifest)
  build_baseline_lr.py    #   register the lr baseline as a normal submission
  cleanup_submission.py   #   maintainer removal of one submission (R2 + KV + repo); dry-run default

server/                   # Python package imported by the GitHub Actions
  intake.py               #   validate-pr entrypoint: validate the metadata-only PR
  validation.py           #   structural + numeric validation of a submission
  scoring.py              #   adapter that scores raw predictions via eval.py
  truth.py                #   download + manifest-sha verify of the canonical truth table
  objectstore.py          #   R2 / filesystem abstraction (incoming keys)
  metadata.py             #   metadata.yaml schema + status lifecycle (pending → scored)
  archive.py              #   long-term archive abstraction (manual/scheduled sweep)

worker/                   # Cloudflare Worker — the intake relay
  src/index.js            #   presign R2 uploads, rate-limit (KV), open the submission PR
  wrangler.toml           #   non-secret config (repo, R2 endpoint, rate limits, KV binding)

utils/
  plots.py                #   create_html_leaderboard() + matplotlib result plots
  aggregation.py          #   temporal aggregation (weekly/seasonal/anom/iav/site-mean)
  eval_utils.py           #   metric definitions (rmse, mae, nse, …)
  utils.py                #   logging helpers

eval.py                   # load/compare experiments; compute metrics from predictions
reference/                # truth_table.parquet (~164 MB) + committed manifest (cache key)
submissions/              # merged submissions: {model_id}_val_{strategy}/{metadata.yaml + 9 CSVs}
submissions_raw/          # raw lr predictions (source for the truth table)
submissions_metrics/      # preserved MVP metrics (regression check for the lr baseline)
tests/                    # pytest suite for the server package
deploy/                   # SETUP.md, BRINGUP.md, r2-cors.json
.github/workflows/        # validate-pr.yml, score-and-publish.yml
```

---

## Architecture

The leaderboard is a **static site** (`docs/`, served by GitHub Pages) plus a **serverless
intake pipeline** that turns an upload into a merged, scored submission with no VM and no manual
step on the happy path.

### Submission flow

```
 submit.html ── 9 raw CSVs ──► R2 (incoming/)          [browser → presigned PUT, never via Worker]
      │                            ▲
      └── finalize ──► Worker ─────┘  opens same-repo PR adding
                                       submissions/{id}_val_{strategy}/metadata.yaml  (R2 pointers only)
                                              │
                          validate-pr.yml ◄───┘  download raw from R2 → full validation
                                              │   → comment result → AUTO-MERGE on pass
                                              ▼
                          push to main ──► score-and-publish.yml
                                              │   score pending (join lr truth + eval.py)
                                              │   write 9 metric CSVs, flip metadata → scored
                                              │   rebuild docs/, commit "[skip ci]"
                                              ▼
                                       GitHub Pages redeploys docs/
```

1. **Upload.** `docs/submit.html` uploads the 9 raw prediction CSVs straight to **Cloudflare R2**
   via presigned multipart URLs. The **Worker** ([worker/src/index.js](worker/src/index.js)) only
   presigns and rate-limits (KV) — file bytes never pass through it.
2. **PR.** On finalize, the Worker opens a PR **in this repo** adding a single
   `submissions/{model_id}_val_{strategy}/metadata.yaml` that points at the R2 keys. It never
   commits prediction data or code. Opening from a same-repo branch is what lets the validation
   Action see the secrets + write token it needs.
3. **Validate + merge.** [validate-pr.yml](.github/workflows/validate-pr.yml) downloads the raw
   files from R2, runs the repo's validator on the *data* (never PR-supplied code), comments
   pass/fail, and **squash-auto-merges** on pass. Fork PRs get a read-only token and no secrets,
   so they can't bypass the relay.
4. **Score + publish.** Merging is a push to `main`, which triggers
   [score-and-publish.yml](.github/workflows/score-and-publish.yml): it scores every `pending`
   submission, rebuilds `docs/`, and commits the metrics + `docs/` + `scored` status back. The
   `[skip ci]` in that commit message is the loop guard (its own commit also touches a
   `metadata.yaml`).

### Truth model

The trusted **`lr` baseline's raw predictions** define ground truth. `build_truth_table.py` writes
`reference/truth_table.parquet` keyed by `(setting, target, site_id, time) → y_true`, plus a JSON
manifest whose content hash is the CI cache key. At scoring time a submitter's `y_true` is
**discarded** — only their `y_pred` is scored against the canonical truth via `eval.py`. The truth
table also defines the **required index** (the rows a complete submission must cover).

> Pin note: `pyarrow` is held `<20` across all consumers because ≥20 writes column-chunk size
> statistics that older readers reject. Bump every consumer together if you move it.

---

## Submission format (reference)

A submission is **9 metric CSVs** = 3 settings × 3 targets, for one `val_strategy`, sitting beside
the `metadata.yaml` the relay adds:

```
submissions/{model_id}_val_{val_strategy}/
  metadata.yaml
  {setting}_{target}_{model_id}_val_{val_strategy}.csv      # × 9
```

| field | valid values |
|---|---|
| `setting` | `time-split`, `spatial-easy40`, `TA40` |
| `target` | `GPP`, `ET`, `NEE` |
| `val_strategy` | `mean`, `max`, `discrepancy` |

CSV columns (exact order):
`target,setting,model,scale,env,n_samples,mse,rmse,mae,nse,r2_score,bias,relative_mae,relative_bias`

The leaderboard shows **RMSE** by temporal scale (hourly → iav) and a **Skill score** summary
relative to the `lr` baseline (higher = better).

---

## Ops

- **Rebuild the truth table:** `python scripts/build_truth_table.py` (after changing the `lr` raw set),
  then `python scripts/build_baseline_lr.py` to re-register + regression-check the baseline.
- **Remove a submission:** `python scripts/cleanup_submission.py` (dry-run by default; clears R2,
  KV owner key, and the repo folder).
- **Deploy / first-time setup:** see [deploy/SETUP.md](deploy/SETUP.md) and
  [deploy/BRINGUP.md](deploy/BRINGUP.md). Worker secrets via `wrangler secret put`; Pages serves
  `/docs` on `main`.
