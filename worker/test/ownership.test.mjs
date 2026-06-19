// Ownership-boundary tests for the intake Worker.
//
// Ownership key is the submitter's EMAIL: the stored record is sha256(normalized email).
// Focus: the create-time gate that stops an outsider from overwriting an already-claimed
// model's staged R2 files, plus the finalize backstop + single-write registration.
//
// Run: `npm test` (from worker/) or `node --test test/ownership.test.mjs`.
// No network: globalThis.fetch is stubbed; KV is a Map.

import { test } from "node:test";
import assert from "node:assert/strict";
import { createMultipart, finalize, sha256hex } from "../src/index.js";

const SETTINGS = ["time-split", "spatial-easy40", "TA40"];
const TARGETS = ["GPP", "ET", "NEE"];

// --- fakes ------------------------------------------------------------------

function fakeEnv(seed = {}) {
  const store = new Map(Object.entries(seed));
  const writes = []; // every RL.put, for "written once" assertions
  return {
    RL: {
      get: async (k) => (store.has(k) ? store.get(k) : null),
      put: async (k, v) => { writes.push(k); store.set(k, v); },
    },
    _store: store,
    _writes: writes,
    R2_ENDPOINT: "https://r2.example.com",
    R2_BUCKET: "fluxtrapolation",
    R2_ACCESS_KEY_ID: "test-key",
    R2_SECRET_ACCESS_KEY: "test-secret",
    RL_CREATES_PER_HOUR: "100",
    RL_SUBMISSIONS_PER_HOUR: "100",
    RL_SUBMISSIONS_PER_DAY: "100",
    GITHUB_REPO: "owner/repo",
    GITHUB_TOKEN: "gh-token",
    ALLOWED_ORIGIN: "*",
  };
}

const fakeReq = { headers: { get: (k) => (k === "CF-Connecting-IP" ? "1.2.3.4" : null) } };

const incomingKeys = (modelId, strategy) =>
  SETTINGS.flatMap((s) => TARGETS.map((t) =>
    `incoming/${modelId}_val_${strategy}/${s}_${t}_${modelId}_val_${strategy}_predictions.csv`));

// Install a global fetch stub that records calls and answers R2 + GitHub.
// Returns { calls, restore }. `r2Writes()` counts calls that actually mutate R2 (multipart create).
function installFetch({ listModel = "m2", listStrategy = "mean" } = {}) {
  const calls = [];
  const prev = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    const url = typeof input === "string" || input instanceof URL ? String(input) : input.url;
    const method = (input && input.method) || (init && init.method) || "GET";
    calls.push({ url, method });
    if (url.includes("?uploads") || url.endsWith("uploads")) // R2 InitiateMultipartUpload
      return new Response(
        "<InitiateMultipartUploadResult><UploadId>UP123</UploadId></InitiateMultipartUploadResult>",
        { status: 200 });
    if (url.includes("list-type=2")) { // R2 ListObjectsV2
      const body = "<ListBucketResult>" +
        incomingKeys(listModel, listStrategy).map((k) => `<Contents><Key>${k}</Key></Contents>`).join("") +
        "</ListBucketResult>";
      return new Response(body, { status: 200 });
    }
    if (url.includes("api.github.com")) {
      if (url.includes("/git/ref/heads/main")) return new Response(JSON.stringify({ object: { sha: "base" } }), { status: 200 });
      if (url.includes("/git/refs")) return new Response("{}", { status: 201 });
      if (url.includes("/contents/") && method === "GET") return new Response(JSON.stringify({ message: "Not Found" }), { status: 404 });
      if (url.includes("/contents/")) return new Response(JSON.stringify({ content: {} }), { status: 200 });
      if (url.includes("/pulls")) return new Response(JSON.stringify({ html_url: "https://github.com/owner/repo/pull/1" }), { status: 201 });
    }
    return new Response("{}", { status: 200 });
  };
  return {
    calls,
    r2Writes: () => calls.filter((c) => c.url.includes("?uploads") || c.url.endsWith("uploads")).length,
    prCreated: () => calls.some((c) => c.url.includes("/pulls")),
    restore: () => { globalThis.fetch = prev; },
  };
}

const fn = (modelId, strategy = "mean", setting = "time-split", target = "GPP") =>
  `${setting}_${target}_${modelId}_val_${strategy}_predictions.csv`;

async function callCreate(env, body) {
  const res = await createMultipart(env, body, fakeReq);
  return { status: res.status, body: await res.json() };
}
async function callFinalize(env, body) {
  const res = await finalize(env, body, fakeReq);
  return { status: res.status, body: await res.json() };
}

// --- create: the new pre-upload gate ----------------------------------------

test("create on an existing model with the WRONG email -> 403, no R2 write", async () => {
  const env = fakeEnv({ "owner:m1": await sha256hex("right@example.com") });
  const f = installFetch();
  try {
    const { status } = await callCreate(env, { model_id: "m1", val_strategy: "mean", filename: fn("m1"), email: "wrong@example.com" });
    assert.equal(status, 403);
    assert.equal(f.r2Writes(), 0, "no presigned upload should be created");
    assert.equal(env._writes.length, 0, "no KV write on a rejected create");
  } finally { f.restore(); }
});

test("create with a MISSING or invalid email -> 400, no R2 write", async () => {
  const env = fakeEnv({ "owner:m1": await sha256hex("right@example.com") });
  const f = installFetch();
  try {
    for (const email of ["", "   ", null, undefined, "notanemail"]) {
      const { status } = await callCreate(env, { model_id: "m1", val_strategy: "mean", filename: fn("m1"), email });
      assert.equal(status, 400, `invalid email ${JSON.stringify(email)} must be rejected`);
    }
    assert.equal(f.r2Writes(), 0, "no presigned upload for any invalid-email attempt");
  } finally { f.restore(); }
});

test("create on an existing model with the CORRECT email -> allowed (case-insensitive)", async () => {
  const env = fakeEnv({ "owner:m1": await sha256hex("right@example.com") });
  const f = installFetch();
  try {
    const { status, body } = await callCreate(env, { model_id: "m1", val_strategy: "mean", filename: fn("m1"), email: "RIGHT@Example.com" });
    assert.equal(status, 200);
    assert.equal(body.uploadId, "UP123");
    assert.match(body.key, /^incoming\/m1_val_mean\//);
    assert.equal(f.r2Writes(), 1, "exactly one presigned upload created");
    assert.equal(env._writes.filter((k) => k.startsWith("owner:")).length, 0, "create never writes the ownership record");
  } finally { f.restore(); }
});

test("create for a NEW model_id is allowed with a valid email, and writes no KV record", async () => {
  const env = fakeEnv();
  const f = installFetch();
  try {
    const { status, body } = await callCreate(env, { model_id: "newbie", val_strategy: "mean", filename: fn("newbie"), email: "new@example.com" });
    assert.equal(status, 200);
    assert.equal(body.uploadId, "UP123");
    assert.equal(f.r2Writes(), 1);
    assert.equal(env._store.has("owner:newbie"), false, "registration is deferred to finalize");
  } finally { f.restore(); }
});

// --- finalize: single-write registration + backstop -------------------------

test("finalize for a new model registers owner=sha256(email) once and stashes the raw email", async () => {
  const env = fakeEnv();
  const f = installFetch({ listModel: "m2" });
  try {
    const { status, body } = await callFinalize(env, { model_id: "m2", val_strategy: "mean", display_name: "M2", email: "alice@example.com" });
    assert.equal(status, 200);
    assert.ok(body.prUrl, "a PR url is returned");
    assert.equal(body.ownerToken, undefined, "no owner token is issued anymore");
    const ownerWrites = env._writes.filter((k) => k === "owner:m2");
    assert.equal(ownerWrites.length, 1, "owner record written exactly once");
    assert.equal(env._store.get("owner:m2"), await sha256hex("alice@example.com"), "stored hash is sha256(email)");
    assert.equal(env._store.get("email:m2"), "alice@example.com", "raw email stashed privately for contact");
  } finally { f.restore(); }
});

test("finalize with a missing email -> 400, no PR", async () => {
  const env = fakeEnv();
  const f = installFetch({ listModel: "m2" });
  try {
    const { status } = await callFinalize(env, { model_id: "m2", val_strategy: "mean", display_name: "M2" });
    assert.equal(status, 400);
    assert.equal(f.prCreated(), false, "no PR opened without a valid email");
  } finally { f.restore(); }
});

test("finalize backstop: wrong email on an existing model -> 403, no PR, no overwrite", async () => {
  const recorded = await sha256hex("right@example.com");
  const env = fakeEnv({ "owner:m2": recorded });
  const f = installFetch({ listModel: "m2" });
  try {
    const { status } = await callFinalize(env, { model_id: "m2", val_strategy: "mean", display_name: "M2", email: "wrong@example.com" });
    assert.equal(status, 403);
    assert.equal(f.prCreated(), false, "no PR opened for a rejected finalize");
    assert.equal(env._store.get("owner:m2"), recorded, "ownership record untouched");
  } finally { f.restore(); }
});
