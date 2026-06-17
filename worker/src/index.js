// FLUXtrapolation intake relay (Cloudflare Worker).
//
// Endpoints (called by the Uppy drop box in docs/submit.html):
//   POST /api/multipart/create   -> { key, uploadId }      (start a resumable R2 upload)
//   POST /api/multipart/sign     -> { url }                 (presigned PUT for one part)
//   POST /api/multipart/complete -> { location }            (finish a file)
//   POST /api/multipart/abort    -> {}                      (cancel a file)
//   POST /api/submission/finalize-> { prUrl, ownerToken? }  (verify 9 files, open the PR)
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
const bad = (env, msg, status = 400) => json(env, { error: msg }, status);

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

// KV rate limit: increment a TTL'd counter; eventually-consistent (spam backstop, not a quota).
async function allow(env, key, limit, ttl) {
  const cur = parseInt((await env.RL.get(key)) || "0", 10);
  if (cur >= limit) return false;
  await env.RL.put(key, String(cur + 1), { expirationTtl: ttl });
  return true;
}

// ----------------------------------------------------------------------------- multipart

async function createMultipart(env, body, req) {
  const { model_id, val_strategy, filename } = body;
  if (!MODEL_ID_RE.test(model_id || "")) return bad(env, "invalid model_id");
  if (!VALID_STRATEGIES.includes(val_strategy)) return bad(env, "invalid val_strategy");
  const info = parseFilename(filename || "");
  if (!info || info.model !== model_id || info.strategy !== val_strategy)
    return bad(env, `filename must be {setting}_{target}_${model_id}_val_${val_strategy}.csv`);

  const hour = Math.floor(Date.now() / 3.6e6);
  if (!(await allow(env, `cr:${clientIp(req)}:${hour}`, +env.RL_CREATES_PER_HOUR, 3600)))
    return bad(env, "rate limit: too many upload starts this hour", 429);

  const key = incomingKey(model_id, val_strategy, filename);
  const resp = await r2(env).fetch(objUrl(env, key, "uploads"), { method: "POST" });
  if (!resp.ok) return bad(env, `R2 createMultipart failed: ${resp.status}`, 502);
  const uploadId = xmlTag(await resp.text(), "UploadId");
  if (!uploadId) return bad(env, "R2 returned no UploadId", 502);
  return json(env, { key, uploadId });
}

async function signPart(env, body) {
  const { key, uploadId, partNumber } = body;
  if (!key || !uploadId || !partNumber) return bad(env, "key, uploadId, partNumber required");
  const signed = await r2(env).sign(
    objUrl(env, key, `partNumber=${partNumber}&uploadId=${encodeURIComponent(uploadId)}`),
    { method: "PUT", aws: { signQuery: true } }
  );
  return json(env, { url: signed.url });
}

async function completeMultipart(env, body) {
  const { key, uploadId, parts } = body;
  if (!key || !uploadId || !Array.isArray(parts)) return bad(env, "key, uploadId, parts required");
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
  if (!resp.ok) return bad(env, `R2 complete failed: ${resp.status}`, 502);
  return json(env, { location: objUrl(env, key) });
}

async function abortMultipart(env, body) {
  const { key, uploadId } = body;
  if (!key || !uploadId) return bad(env, "key, uploadId required");
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
  const { model_id, val_strategy, owner_token } = body;
  if (!MODEL_ID_RE.test(model_id || "")) return bad(env, "invalid model_id");
  if (!VALID_STRATEGIES.includes(val_strategy)) return bad(env, "invalid val_strategy");

  const hour = Math.floor(Date.now() / 3.6e6);
  const day = Math.floor(Date.now() / 8.64e7);
  if (!(await allow(env, `fz:${clientIp(req)}:${hour}`, +env.RL_SUBMISSIONS_PER_HOUR, 3600)))
    return bad(env, "rate limit: too many submissions this hour", 429);
  if (!(await allow(env, `fzg:${day}`, +env.RL_SUBMISSIONS_PER_DAY, 86400)))
    return bad(env, "rate limit: global daily submission cap reached", 429);

  // Verify exactly the 9 (setting,target) files are present in R2.
  const keys = await listIncoming(env, model_id, val_strategy);
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
  if (missing.length) return bad(env, `incomplete upload; missing: ${missing.join(", ")}`);

  // Ownership: register on first submission; require matching token on updates.
  const ownerKey = `owner:${model_id}`;
  const recorded = await env.RL.get(ownerKey);
  let issuedToken = null;
  let ownerHash;
  if (recorded) {
    if (!owner_token || (await sha256hex(owner_token)) !== recorded)
      return bad(env, "model_id is owned by another submitter; owner token does not match", 403);
    ownerHash = recorded;
  } else {
    const token = owner_token || crypto.randomUUID();
    ownerHash = await sha256hex(token);
    await env.RL.put(ownerKey, ownerHash); // no TTL: permanent ownership record
    if (!owner_token) issuedToken = token; // return generated token once
  }

  const yamlText = buildMetadataYaml({
    model_id,
    display_name: body.display_name,
    email: body.email,
    description: body.description,
    code_url: body.code_url,
    paper_url: body.paper_url,
    owner: ownerHash,
    val_strategy,
    submitted_at: new Date().toISOString().replace(/\.\d+Z$/, "+00:00"),
    files: files.sort((a, b) => a.filename.localeCompare(b.filename)),
  });

  const prUrl = await openPr(env, model_id, val_strategy, yamlText);
  return json(env, { prUrl, ownerToken: issuedToken });
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
    if (!handler) return bad(env, "not found", 404);
    if (request.method !== "POST") return bad(env, "POST required", 405);
    try {
      const body = await request.json();
      return await handler(env, body, request);
    } catch (e) {
      return bad(env, `error: ${e.message}`, 500);
    }
  },
};
