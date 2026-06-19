// FLUXtrapolation intake relay (Cloudflare Worker).
//
// Endpoints (called by the Uppy drop box in docs/submit.html):
//   POST /api/multipart/create   -> { key, uploadId }      (start a resumable R2 upload)
//   POST /api/multipart/sign     -> { url }                 (presigned PUT for one part)
//   POST /api/multipart/complete -> { location }            (finish a file)
//   POST /api/multipart/abort    -> {}                      (cancel a file)
//   POST /api/submission/finalize-> { prUrl }               (verify 9 files, open the PR)
//
// Security model: the browser uploads directly to R2 via presigned URLs (never through the
// Worker). The Worker holds R2 creds + a fine-grained GitHub token and is the ONLY way a
// submission PR is opened (from a branch in this repo, so the validate-pr Action gets the
// secrets + write token it needs to auto-merge). See deploy/SETUP.md.

import { AwsClient } from "aws4fetch";

const VALID_SETTINGS = ["time-split", "spatial-easy40", "TA40"];
const VALID_TARGETS = ["GPP", "ET", "NEE"];
const VALID_STRATEGIES = ["mean", "max", "discrepancy"];
const MODEL_ID_RE = /^[a-z0-9][a-z0-9_-]{0,48}$/; // lowercase slug
const MAX_PART_SIZE = 64 * 1024 * 1024; // informational; Uppy sets part size client-side

// Shown to a submitter who hits a claimed model_id with a wrong/blank email. The drop box
// keys off the `owner_mismatch` code (403) to render this without an unhelpful "retry" prompt.
const OWNER_MISMATCH_MSG =
  "This model name is already registered to another submitter. To update it, submit again with " +
  "the same email you used on your first submission.";

// ----------------------------------------------------------------------------- helpers

function cors(env) {
  return {
    "Access-Control-Allow-Origin": env.ALLOWED_ORIGIN || "*",
    "Access-Control-Allow-Methods": "GET,POST,DELETE,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
  };
}
const json = (env, data, status = 200) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...cors(env) },
  });
// Every error response carries a machine-stable `code` (for the UI to branch on) plus a
// human-readable `error` string. Keep codes stable; they are part of the Worker's contract.
const bad = (env, code, msg, status = 400) => json(env, { code, error: msg }, status);

function r2(env) {
  return new AwsClient({
    accessKeyId: env.R2_ACCESS_KEY_ID,
    secretAccessKey: env.R2_SECRET_ACCESS_KEY,
    service: "s3",
    region: "auto",
  });
}
function objUrl(env, key, query) {
  const enc = key.split("/").map(encodeURIComponent).join("/");
  return `${env.R2_ENDPOINT}/${env.R2_BUCKET}/${enc}${query ? "?" + query : ""}`;
}

async function sha256hex(s) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
function xmlTag(xml, tag) {
  const m = xml.match(new RegExp(`<${tag}>([^<]*)</${tag}>`));
  return m ? m[1] : null;
}
function xmlTagAll(xml, tag) {
  return [...xml.matchAll(new RegExp(`<${tag}>([^<]*)</${tag}>`, "g"))].map((m) => m[1]);
}

function parseFilename(fn) {
  if (!fn.endsWith(".csv")) return null;
  let base = fn.slice(0, -4);
  if (base.endsWith("_predictions")) base = base.slice(0, -"_predictions".length);
  let strategy = null;
  for (const s of VALID_STRATEGIES)
    if (base.endsWith(`_val_${s}`)) { strategy = s; base = base.slice(0, -`_val_${s}`.length); break; }
  if (!strategy) return null;
  let setting = null;
  for (const s of [...VALID_SETTINGS].sort((a, b) => b.length - a.length))
    if (base.startsWith(`${s}_`)) { setting = s; base = base.slice(`${s}_`.length); break; }
  if (!setting) return null;
  for (const t of VALID_TARGETS)
    if (base.startsWith(`${t}_`)) return { setting, target: t, model: base.slice(`${t}_`.length), strategy };
  return null;
}
const incomingKey = (modelId, strategy, filename) =>
  `incoming/${modelId}_val_${strategy}/${filename}`;

const clientIp = (req) => req.headers.get("CF-Connecting-IP") || "unknown";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

// The submitter's email is the ownership key. Normalize it (trim + lowercase) so the same
// address always hashes the same. Returns null for blank/non-string. Never hash null/blank.
// The RAW email is never written to the public repo — only sha256(email), as `owner`.
function cleanEmail(e) {
  return typeof e === "string" && e.trim() ? e.trim().toLowerCase() : null;
}

// Read-only ownership gate, shared by create (pre-upload) and finalize (backstop).
// Unclaimed model_id -> allowed. Claimed model_id -> allowed only if the email hashes to the
// stored value. Blank/missing/wrong email -> not ok. Never writes KV; registration happens
// once, at finalize.
async function ownerOk(env, modelId, email) {
  const recorded = await env.RL.get(`owner:${modelId}`);
  if (!recorded) return { ok: true, recorded: null };
  const em = cleanEmail(email);
  if (!em || (await sha256hex(em)) !== recorded) return { ok: false, recorded };
  return { ok: true, recorded };
}

// KV rate limit: increment a TTL'd counter; eventually-consistent (spam backstop, not a quota).
async function allow(env, key, limit, ttl) {
  const cur = parseInt((await env.RL.get(key)) || "0", 10);
  if (cur >= limit) return false;
  await env.RL.put(key, String(cur + 1), { expirationTtl: ttl });
  return true;
}

// ----------------------------------------------------------------------------- multipart

async function createMultipart(env, body, req) {
  const { model_id, val_strategy, filename, email } = body;
  if (!MODEL_ID_RE.test(model_id || ""))
    return bad(env, "invalid_model_id", "model_id must be a lowercase slug (a–z, 0–9, _ or -).");
  if (!VALID_STRATEGIES.includes(val_strategy))
    return bad(env, "invalid_val_strategy", "val_strategy must be one of: mean, max, discrepancy.");
  if (!EMAIL_RE.test(cleanEmail(email) || ""))
    return bad(env, "invalid_email", "A valid contact email is required.");
  const info = parseFilename(filename || "");
  if (!info || info.model !== model_id || info.strategy !== val_strategy)
    return bad(env, "invalid_filename",
      `Each file must be named {setting}_{target}_${model_id}_val_${val_strategy}_predictions.csv.`);

  // Ownership gate BEFORE issuing any upload URL: an outsider must never be able to
  // overwrite an already-claimed model's staged R2 files. (Re-checked at finalize.)
  if (!(await ownerOk(env, model_id, email)).ok)
    return bad(env, "owner_mismatch", OWNER_MISMATCH_MSG, 403);

  const hour = Math.floor(Date.now() / 3.6e6);
  if (!(await allow(env, `cr:${clientIp(req)}:${hour}`, +env.RL_CREATES_PER_HOUR, 3600)))
    return bad(env, "rate_limited", "Too many upload starts from your network this hour. Please wait and try again later.", 429);

  const key = incomingKey(model_id, val_strategy, filename);
  const resp = await r2(env).fetch(objUrl(env, key, "uploads"), { method: "POST" });
  if (!resp.ok) return bad(env, "r2_error", `Could not start the upload (storage error ${resp.status}).`, 502);
  const uploadId = xmlTag(await resp.text(), "UploadId");
  if (!uploadId) return bad(env, "r2_error", "Could not start the upload (storage returned no UploadId).", 502);
  return json(env, { key, uploadId });
}

async function signPart(env, body) {
  const { key, uploadId, partNumber } = body;
  if (!key || !uploadId || !partNumber) return bad(env, "bad_request", "key, uploadId, partNumber required.");
  const signed = await r2(env).sign(
    objUrl(env, key, `partNumber=${partNumber}&uploadId=${encodeURIComponent(uploadId)}`),
    { method: "PUT", aws: { signQuery: true } }
  );
  return json(env, { url: signed.url });
}

async function completeMultipart(env, body) {
  const { key, uploadId, parts } = body;
  if (!key || !uploadId || !Array.isArray(parts)) return bad(env, "bad_request", "key, uploadId, parts required.");
  const xml =
    "<CompleteMultipartUpload>" +
    parts
      .sort((a, b) => a.PartNumber - b.PartNumber)
      .map((p) => `<Part><PartNumber>${p.PartNumber}</PartNumber><ETag>${p.ETag}</ETag></Part>`)
      .join("") +
    "</CompleteMultipartUpload>";
  const resp = await r2(env).fetch(objUrl(env, key, `uploadId=${encodeURIComponent(uploadId)}`), {
    method: "POST",
    body: xml,
  });
  if (!resp.ok) return bad(env, "r2_error", `Could not finish the upload (storage error ${resp.status}).`, 502);
  return json(env, { location: objUrl(env, key) });
}

async function abortMultipart(env, body) {
  const { key, uploadId } = body;
  if (!key || !uploadId) return bad(env, "bad_request", "key, uploadId required.");
  await r2(env).fetch(objUrl(env, key, `uploadId=${encodeURIComponent(uploadId)}`), { method: "DELETE" });
  return json(env, {});
}

// ----------------------------------------------------------------------------- finalize

async function listIncoming(env, modelId, strategy) {
  const prefix = `incoming/${modelId}_val_${strategy}/`;
  const url = `${env.R2_ENDPOINT}/${env.R2_BUCKET}?list-type=2&prefix=${encodeURIComponent(prefix)}`;
  const resp = await r2(env).fetch(url);
  if (!resp.ok) throw new Error(`R2 list failed: ${resp.status}`);
  return xmlTagAll(await resp.text(), "Key");
}

function yamlString(v) {
  return v == null ? "null" : JSON.stringify(String(v)); // JSON scalars are valid YAML
}
function buildMetadataYaml(m) {
  const lines = [
    `model_id: ${yamlString(m.model_id)}`,
    `display_name: ${yamlString(m.display_name)}`,
    `email: ${yamlString(m.email)}`,
    `description: ${yamlString(m.description)}`,
    `code_url: ${yamlString(m.code_url)}`,
    `paper_url: ${yamlString(m.paper_url)}`,
    `owner: ${yamlString(m.owner)}`,
    `val_strategy: ${yamlString(m.val_strategy)}`,
    `val_strategy_display: ${yamlString(m.val_strategy_display)}`,
    `submitted_at: ${yamlString(m.submitted_at)}`,
    `is_baseline: false`,
    `status: pending`,
    `files:`,
  ];
  for (const f of m.files) {
    lines.push(`- filename: ${yamlString(f.filename)}`);
    lines.push(`  setting: ${yamlString(f.setting)}`);
    lines.push(`  target: ${yamlString(f.target)}`);
    lines.push(`  r2_key: ${yamlString(f.r2_key)}`);
  }
  return lines.join("\n") + "\n";
}

async function gh(env, method, path, body) {
  const resp = await fetch(`https://api.github.com${path}`, {
    method,
    headers: {
      Authorization: `Bearer ${env.GITHUB_TOKEN}`,
      Accept: "application/vnd.github+json",
      "User-Agent": "fluxtrapolation",
      "Content-Type": "application/json",
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(`GitHub ${method} ${path} -> ${resp.status}: ${JSON.stringify(data)}`);
  return data;
}

async function openPr(env, modelId, strategy, yamlText) {
  const repo = env.GITHUB_REPO;
  const branch = `submission/${modelId}-val-${strategy}-${Date.now()}`;
  const path = `submissions/${modelId}_val_${strategy}/metadata.yaml`;

  const base = await gh(env, "GET", `/repos/${repo}/git/ref/heads/main`);
  await gh(env, "POST", `/repos/${repo}/git/refs`, { ref: `refs/heads/${branch}`, sha: base.object.sha });

  // Create or update the file on the new branch (handles re-submission of an existing model).
  let sha;
  try {
    const ex = await gh(env, "GET", `/repos/${repo}/contents/${path}?ref=${branch}`);
    sha = ex.sha;
  } catch (_) {
    /* new file */
  }
  await gh(env, "PUT", `/repos/${repo}/contents/${path}`, {
    message: `Submission: ${modelId} (val_${strategy})`,
    content: btoa(unescape(encodeURIComponent(yamlText))),
    branch,
    ...(sha ? { sha } : {}),
  });

  const pr = await gh(env, "POST", `/repos/${repo}/pulls`, {
    title: `Submission: ${modelId} (val_${strategy})`,
    head: branch,
    base: "main",
    body: "Automated submission via the drop box. The validate-pr Action will fetch the raw "
      + "files from R2, validate, and auto-merge on pass; scores appear after the VM runs.",
  });
  return pr.html_url;
}

async function finalize(env, body, req) {
  const { model_id, val_strategy, email } = body;
  if (!MODEL_ID_RE.test(model_id || ""))
    return bad(env, "invalid_model_id", "model_id must be a lowercase slug (a–z, 0–9, _ or -).");
  if (!VALID_STRATEGIES.includes(val_strategy))
    return bad(env, "invalid_val_strategy", "val_strategy must be one of: mean, max, discrepancy.");
  if (!EMAIL_RE.test(cleanEmail(email) || ""))
    return bad(env, "invalid_email", "A valid contact email is required.");

  const hour = Math.floor(Date.now() / 3.6e6);
  const day = Math.floor(Date.now() / 8.64e7);
  if (!(await allow(env, `fz:${clientIp(req)}:${hour}`, +env.RL_SUBMISSIONS_PER_HOUR, 3600)))
    return bad(env, "rate_limited", "Too many submissions from your network this hour. Please wait and try again later.", 429);
  if (!(await allow(env, `fzg:${day}`, +env.RL_SUBMISSIONS_PER_DAY, 86400)))
    return bad(env, "rate_limited", "The daily submission limit has been reached. Please try again tomorrow.", 429);

  // Verify exactly the 9 (setting,target) files are present in R2.
  let keys;
  try {
    keys = await listIncoming(env, model_id, val_strategy);
  } catch (_) {
    return bad(env, "r2_error", "Could not read the uploaded files from storage. Please retry.", 502);
  }
  const files = [];
  const combos = new Set();
  for (const key of keys) {
    const info = parseFilename(key.split("/").pop());
    if (!info || info.model !== model_id || info.strategy !== val_strategy) continue;
    combos.add(`${info.setting}/${info.target}`);
    files.push({ filename: key.split("/").pop(), setting: info.setting, target: info.target, r2_key: key });
  }
  const expected = VALID_SETTINGS.flatMap((s) => VALID_TARGETS.map((t) => `${s}/${t}`));
  const missing = expected.filter((c) => !combos.has(c));
  if (missing.length)
    return bad(env, "incomplete_upload", `Incomplete submission — these setting/target files are missing: ${missing.join(", ")}.`);

  // Ownership backstop: the create step already gated this, but re-check here and
  // register on first submission. This is the single, race-free place we WRITE the
  // KV record (one finalize call, vs. Uppy's parallel creates on eventually-consistent KV).
  const ownerKey = `owner:${model_id}`;
  const gate = await ownerOk(env, model_id, email);
  if (!gate.ok)
    return bad(env, "owner_mismatch", OWNER_MISMATCH_MSG, 403);
  let ownerHash = gate.recorded;
  if (!ownerHash) {
    ownerHash = await sha256hex(cleanEmail(email));
    await env.RL.put(ownerKey, ownerHash); // no TTL: permanent ownership record
  }
  // Stash the raw email privately (KV) so the maintainer can contact the submitter.
  // It is NEVER committed to the public repo — only its hash, as `owner`.
  await env.RL.put(`email:${model_id}`, cleanEmail(email));

  const yamlText = buildMetadataYaml({
    model_id,
    display_name: body.display_name,
    email: null,                         // contact email is private (KV), never public
    description: body.description,
    code_url: body.code_url,
    paper_url: null,                     // folded into the "description" / comments field
    owner: ownerHash,
    val_strategy,
    val_strategy_display: body.val_strategy_display,
    submitted_at: new Date().toISOString().replace(/\.\d+Z$/, "+00:00"),
    files: files.sort((a, b) => a.filename.localeCompare(b.filename)),
  });

  let prUrl;
  try {
    prUrl = await openPr(env, model_id, val_strategy, yamlText);
  } catch (_) {
    return bad(env, "github_error", "Your files uploaded, but opening the submission pull request failed. Please retry.", 502);
  }
  return json(env, { prUrl });
}

// ----------------------------------------------------------------------------- router

const ROUTES = {
  "/api/multipart/create": createMultipart,
  "/api/multipart/sign": signPart,
  "/api/multipart/complete": completeMultipart,
  "/api/multipart/abort": abortMultipart,
  "/api/submission/finalize": finalize,
};

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") return new Response(null, { headers: cors(env) });
    const { pathname } = new URL(request.url);
    const handler = ROUTES[pathname];
    if (!handler) return bad(env, "not_found", "not found", 404);
    if (request.method !== "POST") return bad(env, "method_not_allowed", "POST required", 405);
    try {
      const body = await request.json();
      return await handler(env, body, request);
    } catch (e) {
      return bad(env, "server_error", `Unexpected server error: ${e.message}`, 500);
    }
  },
};

// Exported for unit tests (test/ownership.test.mjs); not part of the Worker runtime surface.
export { createMultipart, finalize, ownerOk, cleanEmail, sha256hex, parseFilename };
