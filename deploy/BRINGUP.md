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
Proves: finalize → ownership/KV → GitHub PR creation.
1. Click through finalize (or call `/api/submission/finalize`). Expect a PR URL + (first time) an owner token.
   **Save the token** — `bringup-test` is now a claimed model_id.
2. Open the PR: it must add exactly `submissions/bringup-test_val_mean/metadata.yaml`, `status: pending`,
   9 `files` entries with `r2_key`, `owner:` a sha256 hash (not the raw token).
3. Re-finalize with the wrong owner token → expect HTTP 403 (ownership backstop).
4. **Ownership gate at `create`** (the boundary that protects staged files): now that `bringup-test`
   is claimed, a `create` with a blank/wrong token must be refused *before* any upload URL is issued:
   ```js
   let r = await fetch(W+"/api/multipart/create",{method:"POST",headers:{"Content-Type":"application/json"},
     body:JSON.stringify({model_id:"bringup-test",val_strategy:"mean",
       filename:"time-split_GPP_bringup-test_val_mean_predictions.csv"})});  // no owner_token
   console.log(r.status);  // must be 403; no new object appears in incoming/bringup-test_val_mean/
   ```
   Repeat with the **correct** token → expect 200 + an `uploadId` (a legit owner can still update).

### (d) Action validates + auto-merges
Proves: the sandboxed gate end-to-end. Use **real** lr-style data so validation passes.
1. With valid data uploaded + PR open, watch the `Validate submission` Action.
2. Expect a PR comment with all checks ✅ and an **automatic squash-merge**, no human step.
3. Then submit deliberately broken data (drop 100 rows from one file) → expect the comment to fail
   `index_completeness` and **no merge**.

### (e) Throwaway-fork test — a fork PR cannot merge  ← the security boundary
Proves the claim in the report: forks get a read-only token + no secrets, so they can never merge.
1. From a **throwaway GitHub account**, fork the repo and open a PR adding any
   `submissions/x_val_mean/metadata.yaml` by hand.
2. Expect: the `Validate submission` run shows **no R2 secrets** (the download step fails with
   "R2 config missing"/"could not fetch"), the job **does not auto-merge**, and the merge step
   (if reached) is denied by the read-only token.
3. Confirm the PR stays open/unmerged. This is the test that cannot be done locally — do it once here.

### Cleanup
Delete the `bringup-test` R2 objects, KV `owner:bringup-test`, the throwaway fork PR/branch, and any
test submission folders merged to `main`.
