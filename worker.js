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
const CAPTION_MODEL = '@cf/unum/uform-gen2-qwen-500m'; // fast image-to-text
const EMBED_MODEL = '@cf/baai/bge-base-en-v1.5';       // 768-d text embedding
const DENSE_WEIGHT = 0.55;                              // semantic vs visual mix

// Describe an image as a short English phrase (object type, shape, color,
// material). Returns a trimmed string, or null on failure.
async function getImageCaption(env, imageB64) {
  try {
    const r = await env.AI.run(CAPTION_MODEL, {
      image: imageBytes(imageB64),
      prompt: 'Identify this industrial part or hand tool. Name its type, shape, color and material in one short phrase.',
      max_tokens: 64,
    });
    const txt = (r && (r.description || r.response || r.text)) ||
                (Array.isArray(r) ? (r[0]?.generated_text || r[0]?.description) : '') || '';
    const out = String(txt).replace(/\s+/g, ' ').trim();
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

// Build the combined fingerprint for one image: resnet label vector + caption +
// dense embedding of the caption. Stored as JSON in the items.embedding column.
// Returns { v:2, labels, caption, vec } or null if BOTH signals fail.
async function buildFingerprint(env, imageB64) {
  const labels = await getImageLabels(env, imageB64);
  const caption = await getImageCaption(env, imageB64);
  const vec = caption ? await embedText(env, caption) : null;
  if (!labels && !vec) return null;
  return { v: 2, labels: labels || {}, caption: caption || '', vec: vec || null };
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
      delete row.embedding;
      return json({ ok: true, data: { ...row, has_embedding } });
    }

    // ── Items: list ─────────────────────────────────────────
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

      // has_embedding = item has been AI-indexed (resnet label vector stored).
      const data = (rows.results || []).map(r => {
        const has_embedding = !!r.embedding;
        delete r.embedding;
        return { ...r, has_embedding };
      });
      return json({ ok: true, data, total: total?.n ?? 0, page, limit });
    }

    // ── Items: add ──────────────────────────────────────────
    if (path === '/api/items' && method === 'POST') {
      if (!isAdmin(request, env)) return err('Unauthorized', 401);
      const body = await request.json();
      const { part_code, name, description, category_id, unit, stock_qty, min_stock, image_b64 } = body;
      if (!part_code || !name) return err('part_code และ name จำเป็นต้องกรอก');

      let embeddingJson = null;
      if (image_b64) {
        const fp = await buildFingerprint(env, image_b64);
        if (fp) embeddingJson = JSON.stringify(fp);
      }

      try {
        const res = await env.DB.prepare(`
          INSERT INTO items (part_code, name, description, category_id, unit, stock_qty, min_stock, image_b64, embedding)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        `).bind(part_code, name, description || null, category_id || null, unit || 'ชิ้น',
                stock_qty || 0, min_stock || 0, image_b64 || null, embeddingJson).run();

        if (stock_qty > 0) {
          await env.DB.prepare(`
            INSERT INTO stock_movements (item_id, movement, qty, balance, note, created_by)
            VALUES (?, 'in', ?, ?, 'ยอดเปิด', 'admin')
          `).bind(res.meta.last_row_id, stock_qty, stock_qty).run();
        }
        return json({ ok: true, id: res.meta.last_row_id });
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
      const { name, description, category_id, unit, min_stock, image_b64 } = body;

      const sets = ['name=?', 'description=?', 'category_id=?', 'unit=?', 'min_stock=?', "updated_at=datetime('now','localtime')"];
      const vals = [name, description, category_id, unit, min_stock];
      if (image_b64) {
        const fp = await buildFingerprint(env, image_b64);
        sets.push('image_b64=?', 'embedding=?');
        vals.push(image_b64, fp ? JSON.stringify(fp) : null);
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

    // ── AI Image Match ───────────────────────────────────────
    if (path === '/api/match' && method === 'POST') {
      const body = await request.json();
      const { image_b64, top_k = 5 } = body;
      if (!image_b64) return err('image_b64 จำเป็น');

      // Fingerprint the query photo (visual + semantic), then ensemble-match.
      const queryFp = await buildFingerprint(env, image_b64);
      if (!queryFp) return err('ไม่สามารถวิเคราะห์รูปได้ — ตรวจสอบ Workers AI binding', 502);

      const rows = await env.DB.prepare(
        `SELECT id, part_code, name, description, unit, stock_qty, image_b64, embedding, category_id FROM items WHERE is_active=1 AND embedding IS NOT NULL`
      ).all();
      const items = rows.results || [];
      if (!items.length) {
        return json({ ok: true, matches: [], message: 'ยังไม่มีอุปกรณ์ที่ทำดัชนี AI แล้ว — กดปุ่มทำดัชนีในหน้าแอดมิน' });
      }

      // Base score: similarity to each item's canonical (admin) photo.
      const scored = items.map(it => {
        return { ...it, score: fingerprintScore(queryFp, parseFingerprint(it.embedding)), learned: false };
      });

      // ===== Machine-learning boost (online k-NN over confirmed photos) =====
      // Every confirmed user scan is an extra reference photo. An item's score
      // becomes the best similarity to ANY of its reference photos (canonical
      // OR user-confirmed), so matching gets sharper the more people use it.
      try {
        const exRows = await env.DB.prepare(
          `SELECT item_id, labels, vec FROM match_examples`
        ).all();
        const bestByItem = {};
        for (const ex of exRows.results || []) {
          let labels = {}, vec = null;
          try { labels = ex.labels ? JSON.parse(ex.labels) : {}; } catch {}
          try { vec = ex.vec ? JSON.parse(ex.vec) : null; } catch {}
          const s = fingerprintScore(queryFp, { labels, vec });
          if (bestByItem[ex.item_id] == null || s > bestByItem[ex.item_id]) bestByItem[ex.item_id] = s;
        }
        for (const row of scored) {
          const exb = bestByItem[row.id];
          if (exb != null && exb > row.score) { row.score = exb; row.learned = true; }
        }
      } catch (e) { console.error('Example boost error:', e); }

      scored.sort((a, b) => b.score - a.score);

      const matches = scored.slice(0, top_k).map(m => ({
        id: m.id,
        part_code: m.part_code,
        name: m.name,
        description: m.description,
        unit: m.unit,
        stock_qty: m.stock_qty,
        image_b64: m.image_b64,
        confidence: Math.round(m.score * 100),
        learned: m.learned,
      }));

      return json({ ok: true, matches });
    }

    // ── ML feedback: record a user-confirmed scan as a training example ──
    // Called when a user scans a photo and confirms which item it is (e.g. on
    // requisition submit). Stores the photo's fingerprint as an extra reference
    // for that item so future matches improve. Capped per item to stay bounded.
    if (path === '/api/match/feedback' && method === 'POST') {
      const { image_b64, item_id } = await request.json().catch(() => ({}));
      if (!image_b64 || !item_id) return err('image_b64 และ item_id จำเป็น');
      const item = await env.DB.prepare('SELECT id FROM items WHERE id=? AND is_active=1').bind(item_id).first();
      if (!item) return err('ไม่พบอุปกรณ์', 404);

      const fp = await buildFingerprint(env, image_b64);
      if (!fp) return err('วิเคราะห์รูปไม่สำเร็จ', 502);

      await env.DB.prepare(
        `INSERT INTO match_examples (item_id, labels, vec, caption) VALUES (?, ?, ?, ?)`
      ).bind(item_id, JSON.stringify(fp.labels || {}), fp.vec ? JSON.stringify(fp.vec) : null, fp.caption || '').run();

      // Keep only the most recent 30 examples per item (bounds match-time cost).
      await env.DB.prepare(
        `DELETE FROM match_examples WHERE item_id=? AND id NOT IN (
           SELECT id FROM match_examples WHERE item_id=? ORDER BY id DESC LIMIT 30
         )`
      ).bind(item_id, item_id).run();

      const cnt = await env.DB.prepare('SELECT COUNT(*) n FROM match_examples WHERE item_id=?').bind(item_id).first();
      return json({ ok: true, learned: true, examples: cnt?.n || 0 });
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
      if (!['added', 'linked', 'dismissed'].includes(status)) return err('status ไม่ถูกต้อง');

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
        `SELECT id, image_b64 FROM items WHERE is_active=1 AND image_b64 IS NOT NULL`
      ).all();
      let indexed = 0, failed = 0, semantic = 0;
      for (const it of rows.results || []) {
        const fp = await buildFingerprint(env, it.image_b64);
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
          SELECT ri.*, i.name as item_name, i.part_code, i.unit, i.image_b64
          FROM requisition_items ri JOIN items i ON ri.item_id = i.id
          WHERE ri.req_id = ?
        `).bind(req.id).all();
        return { ...req, items: items.results || [] };
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
      return json({
        ok: true,
        items: items?.n ?? 0,
        requisitions: reqs?.n ?? 0,
        pending: pending?.n ?? 0,
        low_stock: lowStock?.n ?? 0,
        learned_examples: learned?.n ?? 0,
        open_reports: reports?.n ?? 0,
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
