# Intake setup — secrets & one-time configuration

Work top to bottom. Nothing here goes in the repo: Worker secrets live in the Worker's secret
store, Action secrets in GitHub repo settings. The drop box and Worker config hold only
non-secret placeholders.

## Inventory — every secret/value, where it's created, where it's pasted

| Value | Create it at | Paste it into |
|---|---|---|
| **R2 bucket** `fluxtrapolation` | Cloudflare → R2 → Create bucket | `worker/wrangler.toml` `R2_BUCKET`; GitHub secret `R2_BUCKET` |
| **Account ID** (→ endpoint `https://<acct>.r2.cloudflarestorage.com`) | Cloudflare dashboard (right sidebar) | `worker/wrangler.toml` `R2_ENDPOINT`; GitHub secret `R2_ENDPOINT` |
| **R2 access key id + secret** | Cloudflare → R2 → Manage R2 API Tokens → Create (permission: **Object Read & Write**, scoped to the bucket) | Worker secrets `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY`; GitHub secrets of the same names |
| **KV namespace id** | `$(npm config get prefix)/bin/wrangler kv namespace create RL` | `worker/wrangler.toml` `kv_namespaces.id` |
| **GitHub write token** (Worker opens PRs) | GitHub → Settings → Developer settings → **Fine-grained PAT**, repo = `anyafries/FLUXtrapolation-leaderboard`, permissions: **Contents: Read/Write** + **Pull requests: Read/Write** (nothing else) | Worker secret `GITHUB_TOKEN` |
| **TRUTH_TABLE_URL** | Upload `reference/truth_table.parquet` as a GitHub Release asset (or an R2 object) → copy its URL | GitHub secret `TRUTH_TABLE_URL` |
| **Worker URL** (after deploy) | `wrangler deploy` prints it | `docs/submit.html` `WORKER_URL` |

> Both Actions need `R2_ACCESS_KEY_ID/SECRET/ENDPOINT/BUCKET` (to download submissions) and
> `TRUTH_TABLE_URL` (to fetch + verify the truth table): **validate-pr.yml** (pre-merge gate) and
> **score-and-publish.yml** (post-merge scoring) read the same set. The Worker's `GITHUB_TOKEN` is
> what makes submission PRs *same-repo* (so validate-pr gets secrets + a write token and can
> auto-merge); keep its scopes minimal — it is the gatekeeper. score-and-publish uses the built-in
> `GITHUB_TOKEN` with `contents: write` to commit the scored metrics + docs/ (no extra secret).

## Steps

1. **Create the R2 bucket** `fluxtrapolation`.
2. **Apply CORS** so the browser can PUT parts and read each part's `ETag`:
   `$(npm config get prefix)/bin/wrangler r2 bucket cors put fluxtrapolation --rules ./deploy/r2-cors.json`
   (or paste `deploy/r2-cors.json` in the bucket's Settings → CORS). Update `AllowedOrigins` if your
   Pages origin differs.
3. **Lifecycle rule** (orphan hygiene): expire *incomplete multipart uploads* after ~7 days (sweeps
   abandoned browser uploads). **Do NOT** put a short expiry on completed `incoming/` objects:
   under the current architecture **scoring does not delete them** and archiving is a separate
   sweep, so a short timer would erase raw files before they're archived. Lengthen or remove the
   `incoming/` expiry to match your archive cadence — see “Scoring & archiving” below. (You manage
   this rule.)
4. **Create the KV namespace**: `$(npm config get prefix)/bin/wrangler kv namespace create RL` → put the printed id in `wrangler.toml`.
5. **Set Worker secrets** (from `worker/`):
   ```
   $(npm config get prefix)/bin/wrangler secret put R2_ACCESS_KEY_ID
   $(npm config get prefix)/bin/wrangler secret put R2_SECRET_ACCESS_KEY
   $(npm config get prefix)/bin/wrangler secret put GITHUB_TOKEN
   ```
   and edit the `[vars]` in `wrangler.toml` (`R2_ENDPOINT`, `R2_BUCKET`, `ALLOWED_ORIGIN`, `GITHUB_REPO`).
6. **Deploy the Worker**: `cd worker && npm install && $(npm config get prefix)/bin/wrangler deploy`. Copy the printed URL into
   `docs/submit.html` `WORKER_URL`.
7. **Add GitHub Action secrets** (repo → Settings → Secrets and variables → Actions):
   `R2_ENDPOINT`, `R2_BUCKET`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `TRUTH_TABLE_URL`.
8. **Publish the truth table**: upload `reference/truth_table.parquet` as a release asset (or R2 object);
   set `TRUTH_TABLE_URL` to its URL. The Action verifies its sha256 against
   `reference/truth_table_manifest.json` before use.
9. Run the **bring-up test plan** in `deploy/BRINGUP.md` — one layer at a time.

## Scoring & archiving

**Scoring runs in `.github/workflows/score-and-publish.yml`** — there is no VM/poller. When a
submission PR merges (a push to `main`), that workflow scores every still-`pending` submission on
GitHub-hosted runners (eval.py on 9 files takes minutes): it downloads the raw files from R2, joins
the lr truth table, writes the metric CSVs, flips the metadata to `status: scored`, rebuilds the
leaderboard, and commits everything back to `main` (Pages redeploys). Its commit message contains
`[skip ci]` so the push can't re-trigger scoring. It reuses the same R2 + `TRUTH_TABLE_URL` secrets
as validate-pr.yml.

**Archiving is a separate manual/scheduled sweep** — scoring intentionally leaves the raw files in
R2 `incoming/` and never touches the archive. To archive (R2 is S3-compatible; export the R2 keys
as `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` first):

```bash
# 1) list raw files still in the dock
aws s3 ls s3://fluxtrapolation/incoming/ --recursive --endpoint-url "$R2_ENDPOINT"
# 2) copy them to your keep-forever archive (another bucket / local mount / Dropbox / …)
aws s3 sync s3://fluxtrapolation/incoming/ s3://flux-archive/incoming/ --endpoint-url "$R2_ENDPOINT"
# 3) ONLY after verifying the copy, optionally delete a swept submission from the dock
aws s3 rm s3://fluxtrapolation/incoming/<model>_val_<strategy>/ --recursive --endpoint-url "$R2_ENDPOINT"
```

Until you run this, raw files accumulate in `incoming/` — hence keep the lifecycle `incoming/`
expiry long (or off) so nothing is deleted before it's archived (step 3 above). A
`server/archive.py` (ArchiveBackend) abstraction exists for a future automated sweep; it is not
wired into any workflow yet.
