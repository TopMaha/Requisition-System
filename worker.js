/**
 * Requisition System - Cloudflare Worker
 * ระบบเบิกอุปกรณ์/อะไหล่โรงงาน ด้วย AI Image Matching
 *
 * Endpoints:
 *  GET    /api/health
 *  GET    /api/categories
 *  GET    /api/items
 *  POST   /api/items          (admin)
 *  PUT    /api/items/:id      (admin)
 *  DELETE /api/items/:id      (admin)
 *  POST   /api/match          - AI image matching
 *  POST   /api/stock/adjust   (admin)
 *  GET    /api/requisitions
 *  POST   /api/requisitions
 *  PUT    /api/requisitions/:id/status  (admin)
 *  GET    /api/stats
 *  POST   /api/admin/verify   (admin)  - server-side login check
 *  GET    /api/employees
 *  POST   /api/employees      (admin)
 *  PUT    /api/employees/:id  (admin)
 *  DELETE /api/employees/:id  (admin)
 *  GET    /api/machines
 *  POST   /api/machines       (admin)
 *  PUT    /api/machines/:id   (admin)
 *  DELETE /api/machines/:id   (admin)
 */

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, X-Admin-Key',
};

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json', ...CORS },
  });
}

function err(msg, status = 400) {
  return json({ ok: false, error: msg }, status);
}

// ===== Decode a data-URL / base64 image into a byte array =====
function imageBytes(imageB64) {
  const base64 = imageB64.replace(/^data:image\/\w+;base64,/, '');
  const binaryStr = atob(base64);
  const bytes = new Uint8Array(binaryStr.length);
  for (let i = 0; i < binaryStr.length; i++) bytes[i] = binaryStr.charCodeAt(i);
  return Array.from(bytes);
}

// ===== R2 image storage helpers =====
// Decode a data-URL into { bytes, contentType }.
function decodeDataUrl(dataUrl) {
  const m = /^data:(image\/[\w.+-]+);base64,(.*)$/s.exec(dataUrl);
  const contentType = m ? m[1] : 'image/jpeg';
  const b64 = m ? m[2] : dataUrl.replace(/^data:[^,]*,/, '');
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return { bytes, contentType };
}
// Upload a data-URL image to R2 under a fresh key; returns the key.
async function putItemImage(env, dataUrl) {
  const { bytes, contentType } = decodeDataUrl(dataUrl);
  const ext = (contentType.split('/')[1] || 'jpg').replace('jpeg', 'jpg');
  const key = `items/${crypto.randomUUID()}.${ext}`;
  await env.BUCKET.put(key, bytes, {
    httpMetadata: { contentType, cacheControl: 'public, max-age=31536000, immutable' },
  });
  return key;
}
// Convert an ArrayBuffer to a base64 data-URL (for re-fingerprinting R2 images).
function bufToDataUrl(buf, contentType = 'image/jpeg') {
  const bytes = new Uint8Array(buf);
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return `data:${contentType};base64,${btoa(bin)}`;
}
// Build an HTTP image Response from an R2 key (preferred) or inline base64.
// Returns null if neither source has data.
async function imageResponse(env, r2key, b64) {
  const cache = 'public, max-age=31536000, immutable';
  if (r2key && env.BUCKET) {
    const obj = await env.BUCKET.get(r2key);
    if (obj) return new Response(obj.body, {
      headers: { 'Content-Type': obj.httpMetadata?.contentType || 'image/jpeg', 'Cache-Control': cache, ...CORS },
    });
  }
  if (b64) {
    const { bytes, contentType } = decodeDataUrl(b64);
    return new Response(bytes, { headers: { 'Content-Type': contentType, 'Cache-Control': cache, ...CORS } });
  }
  return null;
}

// Visual fingerprint of an image as a sparse ImageNet label vector.
// CLIP is gated on this account, so we use @cf/microsoft/resnet-50 (image
// classification) and treat its label->score distribution as the image's
// vector. Matching is then image-to-image cosine over these vectors, which is
// language-independent (no dependence on Thai item names). Returns a
// { label: score } map, or null on failure.
async function getImageLabels(env, imageB64) {
  try {
    const result = await env.AI.run('@cf/microsoft/resnet-50', {
      image: imageBytes(imageB64),
    });
    const arr = Array.isArray(result) ? result : (result?.data || []);
    const map = {};
    for (const o of arr) if (o && o.label != null) map[String(o.label)] = o.score;
    return Object.keys(map).length ? map : null;
  } catch (e) {
    console.error('ResNet label error:', e);
    return null;
  }
}

// Cosine similarity between two sparse { label: score } vectors.
function labelCosine(a, b) {
  if (!a || !b) return 0;
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  let dot = 0, na = 0, nb = 0;
  for (const k of keys) {
    const x = a[k] || 0, y = b[k] || 0;
    dot += x * y; na += x * x; nb += y * y;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

// ===== Second AI signal: image caption -> dense text embedding =====
// resnet-50 alone only yields coarse top-5 ImageNet labels, which is weak for
// look-alike industrial parts. We add a semantic signal: caption the photo with
// an image-to-text model, then embed that caption into a 768-d dense vector.
// The final match score is an ensemble of the dense (semantic) and label
// (visual) cosines, which lifts accuracy past the single-model ceiling.
// NOTE: @cf/unum/uform-gen2-qwen-500m was DEPRECATED 2026-05-30 (AI error 5028),
// which silently killed the semantic signal and tanked match accuracy. llava is
// the accessible replacement (llama-3.2-vision needs a license 'agree'; CLIP is
// account-gated). llava is a 7B model (slower) but far more descriptive.
// llama-3.2-vision focuses on the main foreground object and ignores the
// background far better than llava — critical for real-world photos shot on a
// cluttered workbench. (One-time license accepted via 'agree' on 2026-06-02.)
const CAPTION_MODEL = '@cf/meta/llama-3.2-11b-vision-instruct'; // object-focused VLM
const EMBED_MODEL = '@cf/baai/bge-m3';                 // 1024-d multilingual (TH+EN) embedding
const DENSE_WEIGHT = 0.7;                              // lean on semantic (object) over noisy scene labels

// Describe an image for matching: the object's identity + any stamped markings.
// Transcribing printed part numbers/letters is the strongest cue to tell apart
// look-alike spare parts. Returns a trimmed string, or null on failure.
async function getImageCaption(env, imageB64) {
  try {
    const r = await env.AI.run(CAPTION_MODEL, {
      image: imageBytes(imageB64),
      prompt: 'You are cataloguing a machine spare part. Describe ONLY the single main object in the foreground; completely ignore the background, table, hands and surroundings. Give: its likely name/type, shape, color, material, and any distinctive features. Then TRANSCRIBE EXACTLY (verbatim) any text, numbers, letters or part codes printed or stamped on the object. Be concise; do not describe the scene.',
      max_tokens: 110,
    });
    const txt = (r && (r.description || r.response || r.text)) ||
                (Array.isArray(r) ? (r[0]?.generated_text || r[0]?.description) : '') || '';
    // Strip the model's boilerplate lead-in so the embedding centers on the
    // actual object words (e.g. "The main object in the foreground is a ...").
    let out = String(txt).replace(/\s+/g, ' ').trim();
    out = out.replace(/^(the\s+)?(main\s+|single\s+)?(object|item|subject|part|tool)\b[^,.]*?\b(is|are|appears to be|seems to be|looks like|:)\s+/i, '');
    out = out.replace(/^(a|an|the)\s+/i, '').trim();
    return out || null;
  } catch (e) {
    console.error('Caption error:', e);
    return null;
  }
}

// Embed a text string into a dense float vector. Returns number[] or null.
async function embedText(env, text) {
  if (!text) return null;
  try {
    const r = await env.AI.run(EMBED_MODEL, { text: [text] });
    const v = r?.data?.[0];
    return Array.isArray(v) && v.length ? v : null;
  } catch (e) {
    console.error('Embed error:', e);
    return null;
  }
}

// Cosine similarity between two dense float vectors.
function cosineDense(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

// Perceptual-hash (dHash) similarity: a deterministic VISUAL signal that
// complements the caption embedding. Both hashes are 16 hex chars (64 bits),
// computed identically on the client (canvas) and the backfill script (sharp).
// Returns 0..1 where ~0.5 is random and 1.0 is an identical layout; rescaled so
// only the discriminative range above chance contributes.
function phashScore(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dist = 0;
  for (let i = 0; i < a.length; i++) {
    let x = (parseInt(a[i], 16) ^ parseInt(b[i], 16)) & 0xf;
    while (x) { dist += x & 1; x >>= 1; }
  }
  const sim = 1 - dist / (a.length * 4);
  return Math.max(0, (sim - 0.5) * 2);
}
// pHash is added as a BONUS (never a penalty): a strong visual match lifts the
// score; a weak one contributes nothing. This avoids hurting items whose stored
// product-shot hash differs from a real-world query photo.
const PHASH_BONUS = 0.4;

// items.phash may hold one hash (legacy) or a JSON array of augmented hashes
// (rotations/brightness/crops). Always return an array.
function parsePhashes(raw) {
  if (!raw) return [];
  if (raw[0] === '[') { try { const a = JSON.parse(raw); return Array.isArray(a) ? a.filter(Boolean) : []; } catch { return []; } }
  return [raw];
}
// Best pHash bonus between any query hash and any of the item's stored hashes.
function bestPhash(queryHashes, itemHashes) {
  let best = 0;
  for (const qh of queryHashes) for (const ih of itemHashes) {
    const v = phashScore(qh, ih); if (v > best) best = v;
  }
  return best;
}

// ===== Jina CLIP v2 — a true IMAGE embedding (image→vector, multimodal) =====
// Replaces the lossy caption→text-embedding path when the 'jina' engine is on.
// Embeds a list of inputs, e.g. [{image: dataUrl}] or [{text: '...'}].
// Returns an array of 1024-float vectors (nulls where an input failed).
const JINA_MODEL = 'jina-clip-v2';
async function jinaEmbed(env, inputs) {
  if (!env.JINA_API_KEY) throw new Error('JINA_API_KEY not set');
  const r = await fetch('https://api.jina.ai/v1/embeddings', {
    method: 'POST',
    headers: { 'Authorization': 'Bearer ' + env.JINA_API_KEY, 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: JINA_MODEL, input: inputs }),
  });
  if (!r.ok) throw new Error('Jina ' + r.status + ': ' + (await r.text()).slice(0, 120));
  const data = await r.json();
  const out = new Array(inputs.length).fill(null);
  for (const d of data.data || []) out[d.index] = d.embedding;
  return out;
}

// Simple settings k/v (e.g. match_engine = 'legacy' | 'jina').
async function getSetting(env, key, fallback) {
  try {
    const row = await env.DB.prepare('SELECT value FROM settings WHERE key=?').bind(key).first();
    return row ? row.value : fallback;
  } catch { return fallback; }
}

// Keep at most `limit` of a given vector source per item (bounds match-time cost).
async function capVectors(env, itemId, source, limit = 30) {
  await env.DB.prepare(
    `DELETE FROM item_vectors WHERE item_id=? AND source=? AND id NOT IN (
       SELECT id FROM item_vectors WHERE item_id=? AND source=? ORDER BY id DESC LIMIT ${limit})`
  ).bind(itemId, source, itemId, source).run();
}
// Same idea for legacy fingerprint examples, capped per (item, kind).
async function capExamples(env, itemId, kind, limit = 30) {
  await env.DB.prepare(
    `DELETE FROM match_examples WHERE item_id=? AND COALESCE(kind,'positive')=? AND id NOT IN (
       SELECT id FROM match_examples WHERE item_id=? AND COALESCE(kind,'positive')=? ORDER BY id DESC LIMIT ${limit})`
  ).bind(itemId, kind, itemId, kind).run();
}

// Penalty weight: how strongly a known "this photo is NOT item X" demotes item X
// when a similar photo is scanned again. Subtracted from the positive similarity.
const NEG_PENALTY = 0.5;

// Self-healing schema: create the feedback-log table and add the match_examples
// `kind` column if they're missing, so deploying a new worker.js "just works"
// without a manual `wrangler d1 execute`. Runs once per isolate (cold start).
let _migrated = false;
async function ensureSchema(env) {
  if (_migrated || !env.DB) return;
  try {
    await env.DB.prepare(
      `CREATE TABLE IF NOT EXISTS match_feedback (
         id INTEGER PRIMARY KEY AUTOINCREMENT,
         engine TEXT, chosen_item_id INTEGER, chosen_label TEXT,
         rejected_item_ids TEXT, candidates TEXT, source TEXT, reporter TEXT,
         created_at TEXT DEFAULT (datetime('now','localtime'))
       )`
    ).run();
    await env.DB.prepare('CREATE INDEX IF NOT EXISTS idx_match_feedback_created ON match_feedback(id)').run();
    // Add `kind` to match_examples for legacy negative feedback (no-op if it exists).
    try { await env.DB.prepare("ALTER TABLE match_examples ADD COLUMN kind TEXT DEFAULT 'positive'").run(); } catch {}
    _migrated = true;
  } catch (e) { /* leave _migrated=false so the next request retries the migration */ }
}

// Build the combined fingerprint for one image: resnet label vector + caption +
// dense embedding. Stored as JSON in the items.embedding column.
// `extraText` (the item's name + description) is appended to the caption before
// embedding so a stored item is anchored by its known part number/name; a query
// photo whose caption transcribes that same number then matches strongly.
// Pass no extraText for query/example fingerprints (visual/caption only).
// Returns { v:3, labels, caption, vec } or null if BOTH signals fail.
async function buildFingerprint(env, imageB64, extraText) {
  const labels = await getImageLabels(env, imageB64);
  const caption = await getImageCaption(env, imageB64);
  const semantic = [caption, extraText].filter(Boolean).join('. ');
  const vec = semantic ? await embedText(env, semantic) : null;
  if (!labels && !vec) return null;
  return { v: 3, labels: labels || {}, caption: caption || '', vec: vec || null };
}

// Parse a stored embedding cell into a fingerprint, tolerating the legacy
// format where the whole object was a flat { label: score } map.
function parseFingerprint(raw) {
  if (!raw) return null;
  let o = null;
  try { o = typeof raw === 'string' ? JSON.parse(raw) : raw; } catch { return null; }
  if (!o || typeof o !== 'object') return null;
  if (o.labels || o.v) return { labels: o.labels || {}, vec: o.vec || null, caption: o.caption || '' };
  return { labels: o, vec: null, caption: '' }; // legacy: flat label map
}

// Ensemble score between a query fingerprint and a stored fingerprint.
// Uses both signals when available, else degrades to the visual label cosine.
function fingerprintScore(q, stored) {
  if (!q || !stored) return 0;
  const labelSim = labelCosine(q.labels, stored.labels);
  if (!q.vec || !stored.vec) return labelSim;
  const denseSim = cosineDense(q.vec, stored.vec);
  return DENSE_WEIGHT * denseSim + (1 - DENSE_WEIGHT) * labelSim;
}

// ===== Generate req_no =====
function genReqNo() {
  const d = new Date();
  const yy = d.getFullYear().toString().slice(-2);
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  const rand = Math.floor(Math.random() * 9000) + 1000;
  return `REQ${yy}${mm}${dd}-${rand}`;
}

// ===== Check admin =====
function isAdmin(request, env) {
  const key = request.headers.get('X-Admin-Key') || '';
  return key === (env.ADMIN_PASSWORD || 'admin1234');
}

// ===== Router =====
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;

    if (method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS });
    }

    await ensureSchema(env);

    // ── Health ──────────────────────────────────────────────
    if (path === '/api/health') {
      return json({ ok: true, version: '1.1.0', ts: new Date().toISOString() });
    }

    // ── Admin: verify password (server-side login check) ─────
    // The SPA used to check the admin password purely client-side, which
    // silently drifts from the real server password and makes every admin
    // API call 401. This endpoint lets the login verify against the server.
    if (path === '/api/admin/verify' && method === 'POST') {
      if (!isAdmin(request, env)) return err('รหัสผ่านไม่ถูกต้อง', 401);
      return json({ ok: true });
    }

    // ── Categories ──────────────────────────────────────────
    if (path === '/api/categories' && method === 'GET') {
      const rows = await env.DB.prepare('SELECT * FROM categories ORDER BY name').all();
      return json({ ok: true, data: rows.results });
    }

    // ── Item image (binary, cached) ─────────────────────────
    // Serves the image as real bytes so lists/history/match can carry a tiny
    // URL instead of a fat base64 blob. Reads from D1 today; when R2 is enabled
    // this is the single place to swap to env.BUCKET.get(key). The ?v= version
    // token (item.updated_at) lets us cache hard yet refresh on edit.
    const itemImageMatch = path.match(/^\/api\/items\/(\d+)\/image$/);
    if (itemImageMatch && method === 'GET') {
      const row = await env.DB.prepare('SELECT image_key, image_b64 FROM items WHERE id=?').bind(itemImageMatch[1]).first();
      const resp = row && (await imageResponse(env, row.image_key, row.image_b64));
      return resp || new Response('Not found', { status: 404, headers: CORS });
    }

    // ── Item thumbnail (small) ──────────────────────────────
    // Prefers the stored thumbnail; falls back to the full image so items
    // without a thumb (legacy/pre-thumbnail) still display.
    const itemThumbMatch = path.match(/^\/api\/items\/(\d+)\/thumb$/);
    if (itemThumbMatch && method === 'GET') {
      const row = await env.DB.prepare('SELECT thumb_key, image_key, image_b64 FROM items WHERE id=?').bind(itemThumbMatch[1]).first();
      const resp = row && (
        (await imageResponse(env, row.thumb_key, null)) ||
        (await imageResponse(env, row.image_key, row.image_b64))
      );
      return resp || new Response('Not found', { status: 404, headers: CORS });
    }

    // ── Items: get single ───────────────────────────────────
    const itemSingleMatch = path.match(/^\/api\/items\/(\d+)$/);
    if (itemSingleMatch && method === 'GET') {
      const id = itemSingleMatch[1];
      const row = await env.DB.prepare(`
        SELECT i.*, c.name as category_name
        FROM items i LEFT JOIN categories c ON i.category_id = c.id
        WHERE i.id = ? AND i.is_active = 1
      `).bind(id).first();
      if (!row) return err('ไม่พบอุปกรณ์', 404);
      const has_embedding = !!row.embedding; // AI-indexed (resnet label vector stored)
      const has_image = !!(row.image_key || row.image_b64);
      const v = encodeURIComponent(row.updated_at || '');
      const image_url = has_image ? `/api/items/${row.id}/image?v=${v}` : null;
      const thumb_url = has_image ? `/api/items/${row.id}/thumb?v=${v}` : null;
      delete row.embedding;
      delete row.image_key;
      delete row.thumb_key;
      return json({ ok: true, data: { ...row, has_embedding, has_image, image_url, thumb_url } });
    }

    // ── Items: list ─────────────────────────────────────────
    // ── Items: suggest the next running code (I-0001, I-0002, …) ──
    if (path === '/api/items/next-code' && method === 'GET') {
      const rows = await env.DB.prepare("SELECT part_code FROM items WHERE part_code LIKE 'I-%'").all();
      let max = 0;
      for (const r of rows.results || []) {
        const m = /^I-(\d+)$/.exec(r.part_code);
        if (m) { const n = parseInt(m[1], 10); if (n > max) max = n; }
      }
      return json({ ok: true, next: 'I-' + String(max + 1).padStart(4, '0') });
    }

    if (path === '/api/items' && method === 'GET') {
      const search = url.searchParams.get('q') || '';
      const cat = url.searchParams.get('category') || '';
      const page = parseInt(url.searchParams.get('page') || '1');
      const limit = parseInt(url.searchParams.get('limit') || '50');
      const offset = (page - 1) * limit;

      let sql = `
        SELECT i.*, c.name as category_name
        FROM items i LEFT JOIN categories c ON i.category_id = c.id
        WHERE i.is_active = 1
      `;
      const params = [];
      if (search) { sql += ` AND (i.name LIKE ? OR i.part_code LIKE ?)`; params.push(`%${search}%`, `%${search}%`); }
      if (cat)    { sql += ` AND i.category_id = ?`; params.push(cat); }
      sql += ` ORDER BY i.name LIMIT ? OFFSET ?`;
      params.push(limit, offset);

      const rows = await env.DB.prepare(sql).bind(...params).all();
      const total = await env.DB.prepare(
        `SELECT COUNT(*) as n FROM items WHERE is_active=1 ${search ? 'AND (name LIKE ? OR part_code LIKE ?)' : ''}`
      ).bind(...(search ? [`%${search}%`, `%${search}%`] : [])).first();

      // Strip the heavy fields from the list: send a small image URL instead of
      // base64, so big inventories load fast. has_embedding = AI-indexed.
      const data = (rows.results || []).map(r => {
        const has_embedding = !!r.embedding;
        const has_image = !!(r.image_key || r.image_b64);
        const v = encodeURIComponent(r.updated_at || '');
        const image_url = has_image ? `/api/items/${r.id}/image?v=${v}` : null;
        const thumb_url = has_image ? `/api/items/${r.id}/thumb?v=${v}` : null;
        delete r.embedding;
        delete r.image_b64;
        delete r.image_key;
        delete r.thumb_key;
        return { ...r, has_embedding, has_image, image_url, thumb_url };
      });
      return json({ ok: true, data, total: total?.n ?? 0, page, limit });
    }

    // ── Items: add ──────────────────────────────────────────
    if (path === '/api/items' && method === 'POST') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      const body = await request.json();
      const { part_code, name, description, category_id, unit, stock_qty, min_stock, image_b64, thumb_b64, phash, phashes } = body;
      if (!part_code || !name) return err('part_code และ name จำเป็นต้องกรอก');
      const phashStore = Array.isArray(phashes) && phashes.length ? JSON.stringify(phashes) : (phash || null);

      let embeddingJson = null, imageKey = null, thumbKey = null, imageB64Store = null, fp = null;
      if (image_b64) {
        fp = await buildFingerprint(env, image_b64, [name, description].filter(Boolean).join('. '));
        if (fp) embeddingJson = JSON.stringify(fp);
        // Store the image (+thumbnail) in R2; fall back to inline D1 base64.
        if (env.BUCKET) {
          try {
            imageKey = await putItemImage(env, image_b64);
            if (thumb_b64) thumbKey = await putItemImage(env, thumb_b64);
          } catch (e) { console.error('R2 put failed, using D1:', e); imageB64Store = image_b64; imageKey = null; thumbKey = null; }
        } else {
          imageB64Store = image_b64;
        }
      }

      try {
        const res = await env.DB.prepare(`
          INSERT INTO items (part_code, name, description, category_id, unit, stock_qty, min_stock, image_b64, image_key, thumb_key, embedding, phash)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `).bind(part_code, name, description || null, category_id || null, unit || 'ชิ้น',
                stock_qty || 0, min_stock || 0, imageB64Store, imageKey, thumbKey, embeddingJson, phashStore).run();

        const newId = res.meta.last_row_id;
        if (stock_qty > 0) {
          await env.DB.prepare(`
            INSERT INTO stock_movements (item_id, movement, qty, balance, note, created_by)
            VALUES (?, 'in', ?, ?, 'ยอดเปิด', 'admin')
          `).bind(newId, stock_qty, stock_qty).run();
        }

        // Auto-close any pending "not-found" requests this new item satisfies:
        // compare the new item's fingerprint to each งานค้าง report's fingerprint.
        let closedRequests = 0;
        if (fp && fp.vec) {
          try {
            const pend = await env.DB.prepare(
              "SELECT id, labels, vec FROM scan_reports WHERE status='pending_new'"
            ).all();
            for (const p of pend.results || []) {
              let labels = {}, vec = null;
              try { labels = p.labels ? JSON.parse(p.labels) : {}; } catch {}
              try { vec = p.vec ? JSON.parse(p.vec) : null; } catch {}
              const score = fingerprintScore({ labels: fp.labels, vec: fp.vec }, { labels, vec });
              if (score >= 0.72) {
                await env.DB.prepare(
                  "UPDATE scan_reports SET status='added', resolved_item_id=?, resolved_at=datetime('now','localtime') WHERE id=?"
                ).bind(newId, p.id).run();
                // also keep the reported photo as a training example
                await env.DB.prepare(
                  "INSERT INTO match_examples (item_id, labels, vec, caption) VALUES (?, ?, ?, ?)"
                ).bind(newId, p.labels || '{}', p.vec || null, 'auto-closed pending request').run();
                closedRequests++;
              }
            }
          } catch (e) { console.error('auto-close pending error:', e); }
        }
        return json({ ok: true, id: newId, closed_requests: closedRequests });
      } catch (e) {
        if (e.message?.includes('UNIQUE')) return err('รหัสอุปกรณ์นี้มีในระบบแล้ว');
        throw e;
      }
    }

    // ── Items: update ────────────────────────────────────────
    const itemEditMatch = path.match(/^\/api\/items\/(\d+)$/);
    if (itemEditMatch && method === 'PUT') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      const id = itemEditMatch[1];
      const body = await request.json();
      const { name, description, category_id, unit, min_stock, image_b64, thumb_b64, phash, phashes } = body;

      const sets = ['name=?', 'description=?', 'category_id=?', 'unit=?', 'min_stock=?', "updated_at=datetime('now','localtime')"];
      const vals = [name, description, category_id, unit, min_stock];
      const editPhashStore = Array.isArray(phashes) && phashes.length ? JSON.stringify(phashes) : phash;
      if (image_b64 && editPhashStore) { sets.push('phash=?'); vals.push(editPhashStore); }
      if (image_b64) {
        const fp = await buildFingerprint(env, image_b64, [name, description].filter(Boolean).join('. '));
        // Keep the OLD main image's fingerprint as training data (the new photo
        // becomes the main reference; the previous one still helps matching).
        const prev = await env.DB.prepare('SELECT embedding, image_key, thumb_key FROM items WHERE id=?').bind(id).first();
        if (prev?.embedding) {
          try {
            const pf = JSON.parse(prev.embedding);
            await env.DB.prepare('INSERT INTO match_examples (item_id, labels, vec, caption) VALUES (?, ?, ?, ?)')
              .bind(id, JSON.stringify(pf.labels || {}), pf.vec ? JSON.stringify(pf.vec) : null, (pf.caption || '') + ' [previous main image]').run();
          } catch {}
        }
        let imageKey = null, thumbKey = null, imageB64Store = null;
        if (env.BUCKET) {
          try {
            imageKey = await putItemImage(env, image_b64);
            if (thumb_b64) thumbKey = await putItemImage(env, thumb_b64);
          } catch (e) { console.error('R2 put failed, using D1:', e); imageB64Store = image_b64; imageKey = null; thumbKey = null; }
        } else {
          imageB64Store = image_b64;
        }
        // Delete the previous R2 image bytes (the fingerprint is already kept as
        // training data above, so the visual reference is preserved).
        if (imageKey && env.BUCKET) {
          for (const k of [prev?.image_key, prev?.thumb_key]) if (k) { try { await env.BUCKET.delete(k); } catch {} }
        }
        sets.push('image_b64=?', 'image_key=?', 'thumb_key=?', 'embedding=?');
        vals.push(imageB64Store, imageKey, thumbKey, fp ? JSON.stringify(fp) : null);
      }
      vals.push(id);

      await env.DB.prepare(`UPDATE items SET ${sets.join(',')} WHERE id=?`).bind(...vals).run();
      return json({ ok: true });
    }

    // ── Items: delete ────────────────────────────────────────
    if (itemEditMatch && method === 'DELETE') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      const id = itemEditMatch[1];
      await env.DB.prepare(`UPDATE items SET is_active=0 WHERE id=?`).bind(id).run();
      return json({ ok: true });
    }

    // ── Items: reindex ONE item (for bulk re-fingerprinting via a script) ──
    // The bulk /api/reindex loops every item in a single request and would blow
    // the Worker's subrequest/CPU limits on a large catalogue; a driver script
    // calls this per item instead. Rebuilds the fingerprint from the item's R2
    // image + its name/description.
    const itemReindexMatch = path.match(/^\/api\/items\/(\d+)\/reindex$/);
    if (itemReindexMatch && method === 'POST') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      const id = itemReindexMatch[1];
      const row = await env.DB.prepare('SELECT image_b64, image_key, name, description FROM items WHERE id=? AND is_active=1').bind(id).first();
      if (!row) return err('ไม่พบอุปกรณ์', 404);
      let dataUrl = row.image_b64;
      if (!dataUrl && row.image_key && env.BUCKET) {
        const obj = await env.BUCKET.get(row.image_key);
        if (obj) dataUrl = bufToDataUrl(await obj.arrayBuffer(), obj.httpMetadata?.contentType || 'image/jpeg');
      }
      if (!dataUrl) return err('ไม่มีรูปสำหรับทำดัชนี', 400);
      const fp = await buildFingerprint(env, dataUrl, [row.name, row.description].filter(Boolean).join('. '));
      if (!fp) return err('วิเคราะห์รูปไม่สำเร็จ', 502);
      await env.DB.prepare('UPDATE items SET embedding=? WHERE id=?').bind(JSON.stringify(fp), id).run();
      return json({ ok: true, semantic: !!fp.vec, caption: fp.caption });
    }

    // ── Items: set pHash (backfill from a driver script computing it w/ sharp) ──
    const itemPhashMatch = path.match(/^\/api\/items\/(\d+)\/phash$/);
    if (itemPhashMatch && method === 'POST') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      const { phash, phashes } = await request.json().catch(() => ({}));
      const store = Array.isArray(phashes) && phashes.length ? JSON.stringify(phashes) : phash;
      if (!store) return err('phash หรือ phashes จำเป็น');
      await env.DB.prepare('UPDATE items SET phash=? WHERE id=?').bind(store, itemPhashMatch[1]).run();
      return json({ ok: true, count: Array.isArray(phashes) ? phashes.length : 1 });
    }

    // ── Settings (k/v, e.g. match_engine) ────────────────────
    if (path === '/api/settings' && method === 'GET') {
      const rows = await env.DB.prepare('SELECT key, value FROM settings').all();
      const map = {};
      for (const r of rows.results || []) map[r.key] = r.value;
      if (!('match_engine' in map)) map.match_engine = 'legacy';
      map.jina_ready = !!env.JINA_API_KEY;
      return json({ ok: true, settings: map });
    }
    if (path === '/api/settings' && method === 'POST') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      const { key, value } = await request.json().catch(() => ({}));
      if (!key) return err('key จำเป็น');
      await env.DB.prepare('INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value')
        .bind(key, String(value)).run();
      return json({ ok: true });
    }

    // ── Jina: embed an image and store it as a vector for an item ──
    // Driver/feedback calls this with a (client/sharp-)preprocessed image.
    if (path === '/api/jina/embed-store' && method === 'POST') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      const { item_id, image_b64, source, variant } = await request.json().catch(() => ({}));
      if (!item_id || !image_b64) return err('item_id และ image_b64 จำเป็น');
      let vec;
      try { vec = (await jinaEmbed(env, [{ image: image_b64 }]))[0]; }
      catch (e) { return err('Jina embed ล้มเหลว: ' + e.message, 502); }
      if (!vec) return err('Jina ไม่คืนเวกเตอร์', 502);
      await env.DB.prepare('INSERT INTO item_vectors (item_id, vec, source, variant) VALUES (?, ?, ?, ?)')
        .bind(item_id, JSON.stringify(vec), source || 'enroll', variant || 'original').run();
      return json({ ok: true, dims: vec.length });
    }

    // ── Jina: list active items that still have NO vector (for gap-fill) ──
    if (path === '/api/jina/missing' && method === 'GET') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      const rows = await env.DB.prepare(
        'SELECT id FROM items WHERE is_active=1 AND id NOT IN (SELECT DISTINCT item_id FROM item_vectors)'
      ).all();
      return json({ ok: true, ids: (rows.results || []).map(r => r.id) });
    }

    // ── Jina: count vectors per item (for the enroll/eval dashboards) ──
    if (path === '/api/jina/stats' && method === 'GET') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      const tot = await env.DB.prepare('SELECT COUNT(*) n, COUNT(DISTINCT item_id) items FROM item_vectors').first();
      const bySource = await env.DB.prepare('SELECT source, COUNT(*) n FROM item_vectors GROUP BY source').all();
      return json({ ok: true, total: tot?.n || 0, items_with_vectors: tot?.items || 0, by_source: bySource.results || [] });
    }

    // ── AI Image Match ───────────────────────────────────────
    if (path === '/api/match' && method === 'POST') {
      const body = await request.json();
      const { image_b64, top_k = 5 } = body;
      // Accept one hash (phash) or several (phashes: full + centre-crop, etc.).
      const queryHashes = Array.isArray(body.phashes) ? body.phashes.filter(Boolean)
        : (body.phash ? [body.phash] : []);
      if (!image_b64) return err('image_b64 จำเป็น');

      // Engine selection: 'jina' (true image embedding) or 'legacy' (caption
      // ensemble). body.engine overrides the saved setting (for A/B testing).
      const engine = body.engine || await getSetting(env, 'match_engine', 'legacy');
      if (engine === 'jina' && env.JINA_API_KEY) {
        let qvec;
        try { qvec = (await jinaEmbed(env, [{ image: image_b64 }]))[0]; }
        catch (e) { return err('Jina embed ล้มเหลว: ' + e.message, 502); }
        if (!qvec) return err('Jina ไม่คืนเวกเตอร์', 502);
        // Score each Item = MAX cosine over its POSITIVE vectors (enroll/augment/
        // feedback), then SUBTRACT a penalty for its closest NEGATIVE vector
        // (a photo a user explicitly said is NOT this item). Items with only
        // negative vectors never surface (no positive to rank).
        const vrows = await env.DB.prepare('SELECT item_id, vec, source FROM item_vectors').all();
        const posByItem = {}, negByItem = {};
        for (const r of vrows.results || []) {
          let v = null; try { v = JSON.parse(r.vec); } catch {}
          if (!v) continue;
          const s = cosineDense(qvec, v);
          const bucket = r.source === 'negative' ? negByItem : posByItem;
          if (bucket[r.item_id] == null || s > bucket[r.item_id]) bucket[r.item_id] = s;
        }
        const ranked = Object.entries(posByItem)
          .map(([id, pos]) => ({ id: +id, score: pos - NEG_PENALTY * (negByItem[id] || 0) }))
          .sort((a, b) => b.score - a.score).slice(0, top_k);
        if (!ranked.length) return json({ ok: true, matches: [], engine: 'jina', message: 'ยังไม่มีเวกเตอร์ Jina — กดลงทะเบียนในหน้าแอดมิน' });
        const ids = ranked.map(t => t.id);
        const ph = ids.map(() => '?').join(',');
        const detail = await env.DB.prepare(
          `SELECT id, part_code, name, description, unit, stock_qty, min_stock, updated_at,
                  (image_b64 IS NOT NULL OR image_key IS NOT NULL) AS has_image FROM items WHERE id IN (${ph}) AND is_active=1`
        ).bind(...ids).all();
        const byId = {}; for (const d of detail.results || []) byId[d.id] = d;
        const matches = ranked.filter(t => byId[t.id]).map(t => {
          const d = byId[t.id];
          return {
            id: t.id, part_code: d.part_code, name: d.name, description: d.description,
            unit: d.unit, stock_qty: d.stock_qty, min_stock: d.min_stock,
            image_url: d.has_image ? `/api/items/${t.id}/image?v=${encodeURIComponent(d.updated_at || '')}` : null,
            thumb_url: d.has_image ? `/api/items/${t.id}/thumb?v=${encodeURIComponent(d.updated_at || '')}` : null,
            confidence: Math.max(0, Math.min(100, Math.round(t.score * 100))),
          };
        });
        return json({ ok: true, engine: 'jina', matches });
      }

      // Fingerprint the query photo (visual + semantic), then ensemble-match.
      const queryFp = await buildFingerprint(env, image_b64);
      if (!queryFp) return err('ไม่สามารถวิเคราะห์รูปได้ — ตรวจสอบ Workers AI binding', 502);

      // Scan only id + embedding + phash (small) for scoring — NOT the heavy
      // base64 images; we fetch images only for the final winners below.
      const rows = await env.DB.prepare(
        `SELECT id, embedding, phash FROM items WHERE is_active=1 AND embedding IS NOT NULL`
      ).all();
      const items = rows.results || [];
      if (!items.length) {
        return json({ ok: true, matches: [], message: 'ยังไม่มีอุปกรณ์ที่ทำดัชนี AI แล้ว — กดปุ่มทำดัชนีในหน้าแอดมิน' });
      }

      // Base score: caption ensemble, blended with the deterministic pHash
      // visual signal when both the query and the item have a pHash.
      const scored = items.map(it => {
        let s = fingerprintScore(queryFp, parseFingerprint(it.embedding));
        if (queryHashes.length && it.phash) {
          s += PHASH_BONUS * bestPhash(queryHashes, parsePhashes(it.phash));
        }
        return { id: it.id, score: s, learned: false };
      });

      // ===== Machine-learning boost (online k-NN over confirmed photos) =====
      // Every confirmed user scan is an extra reference photo. An item's score
      // becomes the best similarity to ANY of its reference photos (canonical
      // OR user-confirmed), so matching gets sharper the more people use it.
      try {
        const exRows = await env.DB.prepare(
          `SELECT item_id, labels, vec, COALESCE(kind,'positive') AS kind FROM match_examples`
        ).all();
        const posByItem = {}, negByItem = {};
        for (const ex of exRows.results || []) {
          let labels = {}, vec = null;
          try { labels = ex.labels ? JSON.parse(ex.labels) : {}; } catch {}
          try { vec = ex.vec ? JSON.parse(ex.vec) : null; } catch {}
          const s = fingerprintScore(queryFp, { labels, vec });
          const bucket = ex.kind === 'negative' ? negByItem : posByItem;
          if (bucket[ex.item_id] == null || s > bucket[ex.item_id]) bucket[ex.item_id] = s;
        }
        for (const row of scored) {
          // Positive examples lift the score to the best matching reference photo…
          const exb = posByItem[row.id];
          if (exb != null && exb > row.score) { row.score = exb; row.learned = true; }
          // …then a close negative example (user said "not this") demotes it.
          const neg = negByItem[row.id];
          if (neg != null) row.score -= NEG_PENALTY * neg;
        }
      } catch (e) { console.error('Example boost error:', e); }

      scored.sort((a, b) => b.score - a.score);
      const top = scored.slice(0, top_k);
      if (!top.length) return json({ ok: true, matches: [] });

      // Fetch full details (incl. the image) ONLY for the winning few.
      const ids = top.map(t => t.id);
      const ph = ids.map(() => '?').join(',');
      const detailRows = await env.DB.prepare(
        `SELECT id, part_code, name, description, unit, stock_qty, min_stock, updated_at,
                (image_b64 IS NOT NULL OR image_key IS NOT NULL) AS has_image FROM items WHERE id IN (${ph})`
      ).bind(...ids).all();
      const byId = {};
      for (const d of detailRows.results || []) byId[d.id] = d;

      const matches = top.map(t => {
        const d = byId[t.id] || {};
        return {
          id: t.id,
          part_code: d.part_code,
          name: d.name,
          description: d.description,
          unit: d.unit,
          stock_qty: d.stock_qty,
          min_stock: d.min_stock,
          image_url: d.has_image ? `/api/items/${t.id}/image?v=${encodeURIComponent(d.updated_at || '')}` : null,
          thumb_url: d.has_image ? `/api/items/${t.id}/thumb?v=${encodeURIComponent(d.updated_at || '')}` : null,
          confidence: Math.max(0, Math.min(100, Math.round(t.score * 100))),
          learned: t.learned,
        };
      });

      return json({ ok: true, matches });
    }

    // ── ML feedback: record a user-confirmed scan as a training example ──
    // Called when a user scans a photo and confirms which item it is (e.g. on
    // requisition submit). Stores the photo's fingerprint as an extra reference
    // for that item so future matches improve. Capped per item to stay bounded.
    if (path === '/api/match/feedback' && method === 'POST') {
      const body = await request.json().catch(() => ({}));
      const { image_b64, item_id, engine } = body;
      if (!image_b64 || !item_id) return err('image_b64 และ item_id จำเป็น');
      const item = await env.DB.prepare('SELECT id, part_code, name FROM items WHERE id=? AND is_active=1').bind(item_id).first();
      if (!item) return err('ไม่พบอุปกรณ์', 404);

      // Negative-bearing sources — the user explicitly told us the other shown
      // items are WRONG, so penalise them:
      //   manual_pick   = scan flow, rejected all guesses, searched the right item
      //   train_pick    = "สอน ML" page, none of ML's guesses were right
      //   train_confirm = "สอน ML" page, confirmed the correct one among the guesses
      // 'scan' (picked one of the suggestions during a real withdrawal) is
      // positive-only; co-shown items are logged for audit but NOT penalised
      // (they may legitimately look alike and recur together). The source label is
      // kept verbatim in the audit log so training is distinguishable from withdrawals.
      const NEG_SOURCES = new Set(['manual_pick', 'train_pick', 'train_confirm']);
      const KNOWN_SOURCES = new Set(['manual_pick', 'train_pick', 'train_confirm', 'scan']);
      const source = KNOWN_SOURCES.has(body.source) ? body.source : 'scan';
      const negatives = (NEG_SOURCES.has(source) && Array.isArray(body.negatives))
        ? [...new Set(body.negatives.map(Number).filter(n => n && n !== Number(item_id)))] : [];
      const candidates = Array.isArray(body.candidates) ? body.candidates.slice(0, 10) : [];
      const reporter = body.reporter ? String(body.reporter).slice(0, 80) : null;
      const useJina = engine === 'jina' && env.JINA_API_KEY;
      let learned = false, negStored = 0;

      try {
        if (useJina) {
          // Jina: one embed of the scanned photo, reused as a positive for the
          // chosen item and as a negative for each rejected item.
          let v = null;
          try { v = (await jinaEmbed(env, [{ image: image_b64 }]))[0]; } catch (e) { return err('Jina embed ล้มเหลว', 502); }
          if (!v) return err('Jina ไม่คืนเวกเตอร์', 502);
          const vjson = JSON.stringify(v);
          await env.DB.prepare('INSERT INTO item_vectors (item_id, vec, source, variant) VALUES (?, ?, ?, ?)')
            .bind(item_id, vjson, 'feedback', 'scan').run();
          await capVectors(env, item_id, 'feedback');
          learned = true;
          for (const nid of negatives) {
            await env.DB.prepare('INSERT INTO item_vectors (item_id, vec, source, variant) VALUES (?, ?, ?, ?)')
              .bind(nid, vjson, 'negative', 'scan').run();
            await capVectors(env, nid, 'negative');
            negStored++;
          }
        } else {
          // Legacy: build one fingerprint and store as positive / negative examples.
          const fp = await buildFingerprint(env, image_b64);
          if (!fp) return err('วิเคราะห์รูปไม่สำเร็จ', 502);
          const labels = JSON.stringify(fp.labels || {});
          const vec = fp.vec ? JSON.stringify(fp.vec) : null;
          await env.DB.prepare(`INSERT INTO match_examples (item_id, labels, vec, caption, kind) VALUES (?, ?, ?, ?, 'positive')`)
            .bind(item_id, labels, vec, fp.caption || '').run();
          await capExamples(env, item_id, 'positive');
          learned = true;
          for (const nid of negatives) {
            await env.DB.prepare(`INSERT INTO match_examples (item_id, labels, vec, caption, kind) VALUES (?, ?, ?, ?, 'negative')`)
              .bind(nid, labels, vec, fp.caption || '').run();
            await capExamples(env, nid, 'negative');
            negStored++;
          }
        }
      } catch (e) { return err('สอนระบบไม่สำเร็จ: ' + e.message, 502); }

      // Audit log — every feedback event, regardless of engine/source.
      try {
        await env.DB.prepare(
          `INSERT INTO match_feedback (engine, chosen_item_id, chosen_label, rejected_item_ids, candidates, source, reporter)
           VALUES (?, ?, ?, ?, ?, ?, ?)`
        ).bind(
          useJina ? 'jina' : 'legacy', item_id, `${item.name} (${item.part_code})`,
          JSON.stringify(negatives), JSON.stringify(candidates), source, reporter
        ).run();
      } catch (e) { console.error('feedback log error:', e); }

      return json({ ok: true, learned, negatives: negStored, engine: useJina ? 'jina' : 'legacy' });
    }

    // ── ML feedback: audit log (admin) — who taught what, and the AI's guesses ──
    if (path === '/api/match/feedback-log' && method === 'GET') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      const rows = await env.DB.prepare(
        `SELECT id, engine, chosen_item_id, chosen_label, rejected_item_ids, candidates, source, reporter, created_at
         FROM match_feedback ORDER BY id DESC LIMIT 100`
      ).all().catch(() => ({ results: [] }));
      const data = rows.results || [];
      // Resolve rejected item ids → "name (code)" for display.
      const ids = new Set();
      for (const r of data) { try { (JSON.parse(r.rejected_item_ids) || []).forEach(id => ids.add(id)); } catch {} }
      const nameById = {};
      if (ids.size) {
        const arr = [...ids]; const ph = arr.map(() => '?').join(',');
        const nrows = await env.DB.prepare(`SELECT id, part_code, name FROM items WHERE id IN (${ph})`).bind(...arr).all();
        for (const n of nrows.results || []) nameById[n.id] = `${n.name} (${n.part_code})`;
      }
      for (const r of data) {
        let rej = []; try { rej = JSON.parse(r.rejected_item_ids) || []; } catch {}
        r.rejected = rej.map(id => ({ id, label: nameById[id] || ('#' + id) }));
        delete r.rejected_item_ids; delete r.candidates;
      }
      return json({ ok: true, data });
    }

    // ── Scan report: user couldn't find the item -> notify Admin ──
    // Stores the unmatched photo + its fingerprint so Admin can review and
    // either add it as a new item or link it to an existing one (training AI).
    if (path === '/api/scan-reports' && method === 'POST') {
      const { image_b64, note, reporter, top_guess } = await request.json().catch(() => ({}));
      if (!image_b64) return err('image_b64 จำเป็น');
      const fp = await buildFingerprint(env, image_b64); // best-effort; may be null
      const res = await env.DB.prepare(
        `INSERT INTO scan_reports (image_b64, labels, vec, caption, reporter, note, top_guess)
         VALUES (?, ?, ?, ?, ?, ?, ?)`
      ).bind(
        image_b64,
        fp ? JSON.stringify(fp.labels || {}) : null,
        fp && fp.vec ? JSON.stringify(fp.vec) : null,
        fp ? (fp.caption || '') : null,
        reporter || null, note || null, top_guess || null
      ).run();
      return json({ ok: true, id: res.meta.last_row_id });
    }

    // ── Scan reports: list (admin) ──
    if (path === '/api/scan-reports' && method === 'GET') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      const status = url.searchParams.get('status') || 'pending';
      const rows = await env.DB.prepare(
        `SELECT id, image_b64, caption, reporter, note, top_guess, status, resolved_item_id, created_at
         FROM scan_reports WHERE status=? ORDER BY id DESC LIMIT 50`
      ).bind(status).all();
      return json({ ok: true, data: rows.results || [] });
    }

    // ── Scan report: resolve (admin) ──
    // status: 'added' (new item created) | 'linked' (matched existing) | 'dismissed'.
    // When item_id is given, the report's fingerprint is also stored as a
    // training example for that item, so the same photo matches next time.
    const reportMatch = path.match(/^\/api\/scan-reports\/(\d+)\/resolve$/);
    if (reportMatch && method === 'POST') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      const id = reportMatch[1];
      const { status, item_id } = await request.json().catch(() => ({}));
      if (!['added', 'linked', 'dismissed', 'pending_new'].includes(status)) return err('status ไม่ถูกต้อง');

      const rep = await env.DB.prepare('SELECT labels, vec, caption FROM scan_reports WHERE id=?').bind(id).first();
      if (!rep) return err('ไม่พบรายการแจ้ง', 404);

      if (item_id && (status === 'added' || status === 'linked')) {
        const item = await env.DB.prepare('SELECT id FROM items WHERE id=? AND is_active=1').bind(item_id).first();
        if (item) {
          await env.DB.prepare(
            `INSERT INTO match_examples (item_id, labels, vec, caption) VALUES (?, ?, ?, ?)`
          ).bind(item_id, rep.labels || '{}', rep.vec || null, rep.caption || '').run();
        }
      }
      await env.DB.prepare(
        `UPDATE scan_reports SET status=?, resolved_item_id=?, resolved_at=datetime('now','localtime') WHERE id=?`
      ).bind(status, item_id || null, id).run();
      return json({ ok: true });
    }

    // ── Reindex: (re)compute AI label vectors for all items ──
    if (path === '/api/reindex' && method === 'POST') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      const rows = await env.DB.prepare(
        `SELECT id, image_b64, image_key, name, description FROM items WHERE is_active=1 AND (image_b64 IS NOT NULL OR image_key IS NOT NULL)`
      ).all();
      let indexed = 0, failed = 0, semantic = 0;
      for (const it of rows.results || []) {
        // Source the image bytes from D1 (legacy) or R2 (current).
        let dataUrl = it.image_b64;
        if (!dataUrl && it.image_key && env.BUCKET) {
          const obj = await env.BUCKET.get(it.image_key);
          if (obj) dataUrl = bufToDataUrl(await obj.arrayBuffer(), obj.httpMetadata?.contentType || 'image/jpeg');
        }
        if (!dataUrl) { failed++; continue; }
        const fp = await buildFingerprint(env, dataUrl, [it.name, it.description].filter(Boolean).join('. '));
        if (fp) {
          await env.DB.prepare(`UPDATE items SET embedding=? WHERE id=?`)
            .bind(JSON.stringify(fp), it.id).run();
          indexed++;
          if (fp.vec) semantic++;
        } else {
          failed++;
        }
      }
      return json({ ok: true, indexed, semantic, failed, total: (rows.results || []).length });
    }

    // ── AI self-test: confirm caption + embedding models work ──
    if (path === '/api/ai-selftest' && method === 'POST') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      const { image_b64 } = await request.json().catch(() => ({}));
      const out = { caption_model: CAPTION_MODEL, embed_model: EMBED_MODEL };
      // text embedding probe (no image needed)
      const probeVec = await embedText(env, 'steel hex bolt');
      out.embed_ok = !!probeVec;
      out.embed_dims = probeVec ? probeVec.length : 0;
      if (image_b64) {
        const caption = await getImageCaption(env, image_b64);
        out.caption_ok = !!caption;
        out.caption = caption;
        const labels = await getImageLabels(env, image_b64);
        out.label_ok = !!labels;
        out.top_labels = labels ? Object.keys(labels).slice(0, 5) : [];
      }
      return json({ ok: true, ...out });
    }

    // ── Stock Adjust ─────────────────────────────────────────
    if (path === '/api/stock/adjust' && method === 'POST') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      const { item_id, movement, qty, note } = await request.json();
      if (!item_id || !movement || !qty) return err('ข้อมูลไม่ครบ');

      const item = await env.DB.prepare('SELECT stock_qty FROM items WHERE id=?').bind(item_id).first();
      if (!item) return err('ไม่พบอุปกรณ์');

      const newBalance = movement === 'in'
        ? item.stock_qty + qty
        : movement === 'out'
          ? item.stock_qty - qty
          : qty; // adjust = set absolute

      await env.DB.prepare(`UPDATE items SET stock_qty=?, updated_at=datetime('now','localtime') WHERE id=?`)
        .bind(newBalance, item_id).run();
      await env.DB.prepare(`
        INSERT INTO stock_movements (item_id, movement, qty, balance, note, created_by)
        VALUES (?, ?, ?, ?, ?, 'admin')
      `).bind(item_id, movement, qty, newBalance, note || null).run();

      return json({ ok: true, balance: newBalance });
    }

    // ── Requisitions: list ───────────────────────────────────
    if (path === '/api/requisitions' && method === 'GET') {
      const page = parseInt(url.searchParams.get('page') || '1');
      const limit = parseInt(url.searchParams.get('limit') || '20');
      const offset = (page - 1) * limit;
      const status = url.searchParams.get('status') || '';

      let sql = `SELECT * FROM requisitions WHERE 1=1`;
      const params = [];
      if (status) { sql += ` AND status=?`; params.push(status); }
      sql += ` ORDER BY created_at DESC LIMIT ? OFFSET ?`;
      params.push(limit, offset);

      const rows = await env.DB.prepare(sql).bind(...params).all();
      const total = await env.DB.prepare(
        `SELECT COUNT(*) as n FROM requisitions${status ? ' WHERE status=?' : ''}`
      ).bind(...(status ? [status] : [])).first();

      // Load items for each req
      const data = await Promise.all((rows.results || []).map(async req => {
        const items = await env.DB.prepare(`
          SELECT ri.*, i.name as item_name, i.part_code, i.unit, i.updated_at as item_updated_at,
                 (i.image_b64 IS NOT NULL OR i.image_key IS NOT NULL) AS has_image
          FROM requisition_items ri JOIN items i ON ri.item_id = i.id
          WHERE ri.req_id = ?
        `).bind(req.id).all();
        const itemList = (items.results || []).map(it => ({
          ...it,
          image_url: it.has_image ? `/api/items/${it.item_id}/image?v=${encodeURIComponent(it.item_updated_at || '')}` : null,
          thumb_url: it.has_image ? `/api/items/${it.item_id}/thumb?v=${encodeURIComponent(it.item_updated_at || '')}` : null,
        }));
        return { ...req, items: itemList };
      }));

      return json({ ok: true, data, total: total?.n ?? 0, page, limit });
    }

    // ── Requisitions: create ─────────────────────────────────
    if (path === '/api/requisitions' && method === 'POST') {
      const body = await request.json();
      const { requester_name, dept, purpose, items: reqItems } = body;
      if (!requester_name || !reqItems?.length) return err('ข้อมูลไม่ครบ');

      const req_no = genReqNo();
      const res = await env.DB.prepare(`
        INSERT INTO requisitions (req_no, requester_name, dept, purpose)
        VALUES (?, ?, ?, ?)
      `).bind(req_no, requester_name, dept || null, purpose || null).run();

      const req_id = res.meta.last_row_id;
      for (const item of reqItems) {
        await env.DB.prepare(`
          INSERT INTO requisition_items (req_id, item_id, qty_requested, note, match_score)
          VALUES (?, ?, ?, ?, ?)
        `).bind(req_id, item.item_id, item.qty, item.note || null, item.match_score || null).run();
      }

      return json({ ok: true, req_no, id: req_id });
    }

    // ── Requisition: update status ───────────────────────────
    const reqStatusMatch = path.match(/^\/api\/requisitions\/(\d+)\/status$/);
    if (reqStatusMatch && method === 'PUT') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      const id = reqStatusMatch[1];
      const { status, approved_by, note } = await request.json();
      const validStatuses = ['pending', 'approved', 'rejected', 'completed'];
      if (!validStatuses.includes(status)) return err('status ไม่ถูกต้อง');

      await env.DB.prepare(`
        UPDATE requisitions SET status=?, approved_by=?, note=?, updated_at=datetime('now','localtime') WHERE id=?
      `).bind(status, approved_by || null, note || null, id).run();

      // If approved/completed → deduct stock
      if (status === 'completed') {
        const items = await env.DB.prepare(
          `SELECT ri.item_id, ri.qty_requested, i.stock_qty FROM requisition_items ri JOIN items i ON ri.item_id=i.id WHERE ri.req_id=?`
        ).bind(id).all();
        const req = await env.DB.prepare('SELECT req_no FROM requisitions WHERE id=?').bind(id).first();
        for (const item of items.results || []) {
          const newQty = Math.max(0, item.stock_qty - item.qty_requested);
          await env.DB.prepare(`UPDATE items SET stock_qty=? WHERE id=?`).bind(newQty, item.item_id).run();
          await env.DB.prepare(`
            INSERT INTO stock_movements (item_id, req_id, movement, qty, balance, note, created_by)
            VALUES (?, ?, 'out', ?, ?, ?, ?)
          `).bind(item.item_id, id, item.qty_requested, newQty, `เบิก ${req?.req_no}`, approved_by || 'admin').run();
          await env.DB.prepare(`UPDATE requisition_items SET qty_issued=qty_requested WHERE req_id=? AND item_id=?`)
            .bind(id, item.item_id).run();
        }
      }

      return json({ ok: true });
    }

    // ── Stats ────────────────────────────────────────────────
    if (path === '/api/stats' && method === 'GET') {
      const [items, reqs, pending, lowStock, movements, learned] = await Promise.all([
        env.DB.prepare('SELECT COUNT(*) as n FROM items WHERE is_active=1').first(),
        env.DB.prepare('SELECT COUNT(*) as n FROM requisitions').first(),
        env.DB.prepare("SELECT COUNT(*) as n FROM requisitions WHERE status='pending'").first(),
        env.DB.prepare('SELECT COUNT(*) as n FROM items WHERE is_active=1 AND stock_qty <= min_stock').first(),
        env.DB.prepare('SELECT * FROM stock_movements ORDER BY created_at DESC LIMIT 10').all(),
        env.DB.prepare('SELECT COUNT(*) as n FROM match_examples').first().catch(() => ({ n: 0 })),
      ]);
      const reports = await env.DB.prepare("SELECT COUNT(*) as n FROM scan_reports WHERE status='pending'").first().catch(() => ({ n: 0 }));
      const pendingReq = await env.DB.prepare("SELECT COUNT(*) as n FROM scan_reports WHERE status='pending_new'").first().catch(() => ({ n: 0 }));
      return json({
        ok: true,
        items: items?.n ?? 0,
        requisitions: reqs?.n ?? 0,
        pending: pending?.n ?? 0,
        low_stock: lowStock?.n ?? 0,
        learned_examples: learned?.n ?? 0,
        open_reports: reports?.n ?? 0,
        pending_requests: pendingReq?.n ?? 0,
        recent_movements: movements.results || [],
      });
    }

    // ── Employees (master data, shared in D1) ────────────────
    if (path === '/api/employees' && method === 'GET') {
      const rows = await env.DB.prepare('SELECT id, emp_code, emp_name FROM employees ORDER BY emp_code').all();
      return json({ ok: true, data: rows.results || [] });
    }
    if (path === '/api/employees' && method === 'POST') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      const { emp_code, emp_name } = await request.json();
      if (!emp_code || !emp_name) return err('กรุณากรอกรหัสและชื่อพนักงาน');
      try {
        const res = await env.DB.prepare('INSERT INTO employees (emp_code, emp_name) VALUES (?, ?)')
          .bind(emp_code, emp_name).run();
        return json({ ok: true, id: res.meta.last_row_id });
      } catch (e) {
        if (e.message?.includes('UNIQUE')) return err('รหัสพนักงานนี้มีในระบบแล้ว');
        throw e;
      }
    }
    const empMatch = path.match(/^\/api\/employees\/(\d+)$/);
    if (empMatch && method === 'PUT') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      const { emp_code, emp_name } = await request.json();
      if (!emp_code || !emp_name) return err('ข้อมูลไม่ครบ');
      await env.DB.prepare('UPDATE employees SET emp_code=?, emp_name=? WHERE id=?')
        .bind(emp_code, emp_name, empMatch[1]).run();
      return json({ ok: true });
    }
    if (empMatch && method === 'DELETE') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      await env.DB.prepare('DELETE FROM employees WHERE id=?').bind(empMatch[1]).run();
      return json({ ok: true });
    }

    // ── Machines (master data, shared in D1) ─────────────────
    if (path === '/api/machines' && method === 'GET') {
      const rows = await env.DB.prepare('SELECT id, machine_no, zone FROM machines ORDER BY machine_no').all();
      return json({ ok: true, data: rows.results || [] });
    }
    if (path === '/api/machines' && method === 'POST') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      const { machine_no, zone } = await request.json();
      if (!machine_no) return err('กรุณากรอกหมายเลขเครื่องจักร');
      try {
        const res = await env.DB.prepare('INSERT INTO machines (machine_no, zone) VALUES (?, ?)')
          .bind(machine_no, zone || null).run();
        return json({ ok: true, id: res.meta.last_row_id });
      } catch (e) {
        if (e.message?.includes('UNIQUE')) return err('หมายเลขเครื่องนี้มีในระบบแล้ว');
        throw e;
      }
    }
    const machineMatch = path.match(/^\/api\/machines\/(\d+)$/);
    if (machineMatch && method === 'PUT') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      const { machine_no, zone } = await request.json();
      if (!machine_no) return err('ข้อมูลไม่ครบ');
      await env.DB.prepare('UPDATE machines SET machine_no=?, zone=? WHERE id=?')
        .bind(machine_no, zone || null, machineMatch[1]).run();
      return json({ ok: true });
    }
    if (machineMatch && method === 'DELETE') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      await env.DB.prepare('DELETE FROM machines WHERE id=?').bind(machineMatch[1]).run();
      return json({ ok: true });
    }

    return err('Not found', 404);
  },
};
