# Bring-up test plan — isolate one layer at a time

Run these in order. Each step proves one boundary; do not skip ahead to "real submission" — if
something breaks you want to know which layer. Use a throwaway model_id like `bringup-test`
(delete its R2 objects + KV `owner:bringup-test` afterward).

### (a) Presign + CORS — one tiny file from the browser console
Proves: R2 creds, the Worker's presign, and the bucket CORS all line up.
1. Open `https://anyafries.github.io/FLUXtrapolation-leaderboard/submit`, then DevTools console.
2. Create a 1-part multipart upload and sign part 1:
   ```js
   const W = WORKER_URL;
   let c = await (await fetch(W+"/api/multipart/create",{method:"POST",headers:{"Content-Type":"application/json"},
     body:JSON.stringify({model_id:"bringup-test",val_strategy:"mean",
       filename:"time-split_GPP_bringup-test_val_mean_predictions.csv"})})).json();
   let s = await (await fetch(W+"/api/multipart/sign",{method:"POST",headers:{"Content-Type":"application/json"},
     body:JSON.stringify({key:c.key,uploadId:c.uploadId,partNumber:1})})).json();
   let put = await fetch(s.url,{method:"PUT",body:new Blob(["y_true,y_pred,env,site_id,time\n1,1,X,X,2020-01-01 00:00:00\n"])});
   console.log("ETag:", put.headers.get("ETag"));  // must be non-null -> CORS ExposeHeaders OK
   ```
   - **Fail = CORS**: missing `ETag` / blocked PUT → fix `deploy/r2-cors.json` (`AllowedOrigins`, `ExposeHeaders: ETag`).
   - **Fail = presign**: `create`/`sign` 4xx/5xx → check Worker R2 secrets + `R2_ENDPOINT`/`R2_BUCKET`.
3. `complete` it, then confirm the object exists in the R2 bucket browser. 
   ```js
   let done = await (await fetch(W+"/api/multipart/complete",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({key:c.key,uploadId:c.uploadId,parts:[{PartNumber:1,ETag:put.headers.get("ETag")}]})})).json();
   console.log(done);
   ```

### (b) Full 9-file Uppy flow to R2
Proves: the drop box + resumable multipart for real (large) files.
1. Prepare 9 correctly-named CSVs for `bringup-test` (can be small dummies for this step).
2. Use the drop box; watch all 9 reach 100%.
3. Confirm 9 objects under `incoming/bringup-test_val_mean/` in R2. Do **not** finalize yet.

### (c) PR opens with the right metadata.yaml
Proves: finalize → ownership/KV → GitHub PR creation. The ownership key is the submitter's **email**.
1. Click through finalize (or call `/api/submission/finalize`) with an `email`. Expect a PR URL.
   `bringup-test` is now claimed by `sha256(email)` — to update it later, re-submit with the same email.
2. Open the PR: it must add exactly `submissions/bringup-test_val_mean/metadata.yaml`, `status: pending`,
   9 `files` entries with `r2_key`, `owner:` a sha256 hash (the email is **not** stored in the repo).
3. Re-finalize with a different email → expect HTTP 403 (ownership backstop).
4. **Ownership gate at `create`** (the boundary that protects staged files): now that `bringup-test`
   is claimed, a `create` with a missing/wrong email must be refused *before* any upload URL is issued:
   ```js
   let r = await fetch(W+"/api/multipart/create",{method:"POST",headers:{"Content-Type":"application/json"},
     body:JSON.stringify({model_id:"bringup-test",val_strategy:"mean",
       filename:"time-split_GPP_bringup-test_val_mean_predictions.csv",
       email:"someone-else@example.com"})});  // wrong email for a claimed model
   console.log(r.status);  // must be 403; no new object appears in incoming/bringup-test_val_mean/
   ```
   Repeat with the **original** email → expect 200 + an `uploadId` (a legit owner can still update).
5. **Drop-box error UX** (the rejection must read as a rejection): in the drop box, drop the 9
   `bringup-test` files but enter a **different email** than the original, then Submit. Expect the
   result box to turn **amber (⚠)** and show *“This model name is already registered to another
   submitter… submit again with the same email…”* — **not** the red “Upload failed — please retry.”
   The email field should be highlighted/focused, and DevTools console should log
   `… /api/multipart/create -> 403 code=owner_mismatch …`. Then enter the **original** email and
   Submit again → it proceeds. (A 429 rate-limit should likewise show amber with a “wait and try
   later” message; a real network drop is the only case that shows the red “please retry”.)

### (d) Action validates + auto-merges
Proves: the sandboxed gate end-to-end. Use **real** lr-style data so validation passes.
1. With valid data uploaded + PR open, watch the `Validate submission` Action.
2. Expect a PR comment with all checks ✅ and an **automatic squash-merge**, no human step.
3. Then submit deliberately broken data (drop 100 rows from one file) → expect the comment to fail
   `index_completeness` and **no merge**.

### (e) Throwaway-fork test — a fork PR cannot merge  ← the security boundary (TODO!)
Proves the claim in the report: forks get a read-only token + no secrets, so they can never merge.
1. From a **throwaway GitHub account**, fork the repo and open a PR adding any
   `submissions/x_val_mean/metadata.yaml` by hand.
2. Expect: the `Validate submission` run shows **no R2 secrets** (the download step fails with
   "R2 config missing"/"could not fetch"), the job **does not auto-merge**, and the merge step
   (if reached) is denied by the read-only token.
3. Confirm the PR stays open/unmerged. This is the test that cannot be done locally — do it once here.

### (f) Scoring + publish — a merged submission scores and reaches the board
Proves: the post-merge `Score and publish` workflow (no VM). The merge from (d) is the trigger.
1. After (d) auto-merges a **known-good** submission, watch the **`Score and publish`** Action fire
   on the push to `main`.
2. Expect it to: download the raw files from R2, score them, write the 9 metric CSVs into
   `submissions/<model>_val_<strategy>/`, flip that `metadata.yaml` to **`status: scored`** (each
   file gains `sha256`, keeps its `r2_key`), rebuild `docs/`, and push **one** commit titled
   `Score submissions + rebuild leaderboard [skip ci]`.
3. **Loop check:** confirm that scoring commit does **NOT** start another `Score and publish` run
   (the `[skip ci]` guard). There should be exactly one scoring run per merge.
4. Confirm the model appears on the **Pages** leaderboard once Pages redeploys, and that the raw
   files are **still in R2 `incoming/`** (scoring does not delete them; archiving is the separate
   sweep in `deploy/SETUP.md`).
5. **Self-heal check (optional):** open the merged `metadata.yaml`, set `status` back to `pending`,
   commit to `main`; the workflow re-scores it and flips it back to `scored`.

### Cleanup
Delete the `bringup-test` R2 objects, KV `owner:bringup-test` (use `scripts/cleanup_submission.py`),
the throwaway fork PR/branch, and any test submission folders merged to `main`.
