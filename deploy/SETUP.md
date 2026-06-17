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

> The Action also needs `R2_ACCESS_KEY_ID/SECRET/ENDPOINT/BUCKET` (to download submissions) and
> `TRUTH_TABLE_URL` (to fetch + verify the truth table). The Worker's `GITHUB_TOKEN` is what makes
> submission PRs *same-repo* (so the Action gets secrets + a write token and can auto-merge);
> keep its scopes minimal — it is the gatekeeper.

## Steps

1. **Create the R2 bucket** `fluxtrapolation-incoming`.
2. **Apply CORS** so the browser can PUT parts and read each part's `ETag`:
   `wrangler r2 bucket cors put fluxtrapolation-incoming --rules ./deploy/r2-cors.json`
   (or paste `deploy/r2-cors.json` in the bucket's Settings → CORS). Update `AllowedOrigins` if your
   Pages origin differs.
3. **Lifecycle rule** (keeps R2 free): expire incomplete multipart uploads after ~7 days, and expire
   `incoming/` objects after ~7 days (the VM deletes them after scoring; this just sweeps orphans).
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
