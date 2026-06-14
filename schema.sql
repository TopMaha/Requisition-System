-- ===== ระบบเบิกอุปกรณ์/อะไหล่โรงงาน =====
-- Requisition System - D1 Schema

-- ตารางหมวดหมู่อุปกรณ์
CREATE TABLE IF NOT EXISTS categories (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  name      TEXT NOT NULL UNIQUE,
  created_at TEXT DEFAULT (datetime('now','localtime'))
);

-- ตารางอุปกรณ์/อะไหล่
CREATE TABLE IF NOT EXISTS items (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  part_code    TEXT NOT NULL UNIQUE,
  name         TEXT NOT NULL,
  description  TEXT,
  category_id  INTEGER REFERENCES categories(id),
  unit         TEXT DEFAULT 'ชิ้น',
  stock_qty    REAL DEFAULT 0,
  min_stock    REAL DEFAULT 0,
  image_b64    TEXT,              -- legacy inline base64 (new rows use image_key/R2)
  image_key    TEXT,              -- R2 object key (full image); served via /api/items/:id/image
  thumb_key    TEXT,              -- R2 object key (small thumbnail); served via /api/items/:id/thumb
  phash        TEXT,              -- 64-bit dHash (hex) — deterministic visual fingerprint for matching
  embedding    TEXT,              -- JSON array of CLIP float32 vector
  is_active    INTEGER DEFAULT 1,
  created_at   TEXT DEFAULT (datetime('now','localtime')),
  updated_at   TEXT DEFAULT (datetime('now','localtime'))
);

-- ตารางใบเบิก (requisition headers)
CREATE TABLE IF NOT EXISTS requisitions (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  req_no         TEXT NOT NULL UNIQUE,
  requester_name TEXT NOT NULL,
  dept           TEXT,
  purpose        TEXT,
  status         TEXT DEFAULT 'pending',  -- pending | approved | rejected | completed
  approved_by    TEXT,
  note           TEXT,
  created_at     TEXT DEFAULT (datetime('now','localtime')),
  updated_at     TEXT DEFAULT (datetime('now','localtime'))
);

-- ตารางรายการในใบเบิก
CREATE TABLE IF NOT EXISTS requisition_items (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  req_id         INTEGER NOT NULL REFERENCES requisitions(id) ON DELETE CASCADE,
  item_id        INTEGER NOT NULL REFERENCES items(id),
  qty_requested  REAL NOT NULL DEFAULT 1,
  qty_issued     REAL DEFAULT 0,
  note           TEXT,
  match_score    REAL,           -- confidence score จาก image matching
  created_at     TEXT DEFAULT (datetime('now','localtime'))
);

-- ตาราง stock movement log
CREATE TABLE IF NOT EXISTS stock_movements (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  item_id    INTEGER NOT NULL REFERENCES items(id),
  req_id     INTEGER REFERENCES requisitions(id),
  movement   TEXT NOT NULL,      -- 'in' | 'out' | 'adjust'
  qty        REAL NOT NULL,
  balance    REAL NOT NULL,
  note       TEXT,
  created_by TEXT,
  created_at TEXT DEFAULT (datetime('now','localtime'))
);

-- ตารางพนักงาน (master data — รหัสพนักงาน ↔ ชื่อพนักงาน, ใช้ autofill ผู้เบิก)
CREATE TABLE IF NOT EXISTS employees (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  emp_code   TEXT NOT NULL UNIQUE,
  emp_name   TEXT NOT NULL,
  created_at TEXT DEFAULT (datetime('now','localtime'))
);

-- ตารางเครื่องจักร (master data — หมายเลขเครื่อง ↔ zone/พื้นที่)
CREATE TABLE IF NOT EXISTS machines (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  machine_no TEXT NOT NULL UNIQUE,
  zone       TEXT,
  created_at TEXT DEFAULT (datetime('now','localtime'))
);

-- ตารางตัวอย่างที่ผู้ใช้ยืนยัน (machine-learning feedback / online k-NN)
-- ทุกครั้งที่ผู้ใช้สแกนรูปแล้วยืนยันว่าเป็นอุปกรณ์ชิ้นใด เก็บ fingerprint ของรูปนั้น
-- ไว้เป็น "ตัวอย่างอ้างอิงเพิ่มเติม" ของอุปกรณ์ชิ้นนั้น → การ match ครั้งถัดไปแม่นขึ้น
CREATE TABLE IF NOT EXISTS match_examples (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  item_id    INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  labels     TEXT,              -- JSON resnet label map
  vec        TEXT,              -- JSON dense embedding vector
  caption    TEXT,              -- คำบรรยายรูป (debug)
  kind       TEXT DEFAULT 'positive',  -- positive (ยืนยันว่าใช่) | negative (ผู้ใช้บอกว่าไม่ใช่ → ลดคะแนนชิ้นนี้)
  created_at TEXT DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_match_examples_item ON match_examples(item_id);

-- ตารางแจ้ง "ไม่พบอุปกรณ์" (ผู้ใช้สแกนแล้วไม่ตรงกับชิ้นใดในระบบ → แจ้ง Admin)
-- Admin ตรวจสอบ: ถ้าไม่มีจริง → เพิ่มเป็นอุปกรณ์ใหม่ (status=added);
-- ถ้ามีแต่รูปไม่ชัด → ผูกกับชิ้นที่ใช่ (status=linked) แล้วสอน AI ด้วย fingerprint นี้
CREATE TABLE IF NOT EXISTS scan_reports (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  image_b64        TEXT NOT NULL,
  labels           TEXT,           -- JSON resnet label map (fingerprint)
  vec              TEXT,           -- JSON dense embedding (fingerprint)
  caption          TEXT,           -- คำบรรยายรูปจาก AI
  reporter         TEXT,           -- ผู้แจ้ง (รหัส/ชื่อพนักงาน)
  note             TEXT,           -- รายละเอียดที่ผู้ใช้พิมพ์
  top_guess        TEXT,           -- AI เดาว่าอาจเป็นอะไร (context ให้ Admin)
  status           TEXT DEFAULT 'pending',  -- pending | added | linked | dismissed
  resolved_item_id INTEGER REFERENCES items(id),
  created_at       TEXT DEFAULT (datetime('now','localtime')),
  resolved_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_scan_reports_status ON scan_reports(status);

-- ตารางเวกเตอร์ภาพ (Jina CLIP v2) — 1 item มีได้หลาย vector (enroll/augment/feedback/negative)
CREATE TABLE IF NOT EXISTS item_vectors (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  item_id    INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  vec        TEXT NOT NULL,        -- JSON 1024-float Jina CLIP v2 embedding
  source     TEXT DEFAULT 'enroll',-- enroll | augment | feedback | negative
  variant    TEXT,                 -- original | rotate+10 | brightness0.8 | ...
  created_at TEXT DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_item_vectors_item ON item_vectors(item_id);

-- ตาราง log การสอน AI (positive + negative) — ตรวจสอบย้อนหลังได้ว่าใครสอนอะไรเมื่อไหร่
-- ทุกครั้งที่ผู้ใช้ยืนยัน/แก้ผลการค้นหา จะบันทึก: เลือกชิ้นไหน, ปฏิเสธชิ้นใดบ้าง, AI เดาอะไรมา
CREATE TABLE IF NOT EXISTS match_feedback (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  engine            TEXT,           -- jina | legacy
  chosen_item_id    INTEGER REFERENCES items(id),   -- ชิ้นที่ผู้ใช้ยืนยันว่าใช่ (positive)
  chosen_label      TEXT,           -- snapshot ชื่อ+รหัสของชิ้นที่เลือก (ดู log ได้โดยไม่ต้อง join)
  rejected_item_ids TEXT,           -- JSON array ของ item id ที่ AI เสนอแต่ผู้ใช้ไม่เลือก (negatives)
  candidates        TEXT,           -- JSON [{id, confidence}] ภาพรวมที่ AI เสนอตอนนั้น
  source            TEXT,           -- scan (เลือกจากผล AI) | manual_pick (AI เดาผิดทั้งหมด ผู้ใช้ค้นเอง)
  reporter          TEXT,           -- ผู้สอน (รหัส/ชื่อพนักงาน)
  created_at        TEXT DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_match_feedback_created ON match_feedback(id);

-- ตารางตั้งค่าระบบ (key-value) — เก็บ flag เช่น match_engine = legacy | jina
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);

-- seed categories
INSERT OR IGNORE INTO categories (name) VALUES
  ('ชิ้นส่วนเครื่องจักร'),
  ('เครื่องมือช่าง'),
  ('อุปกรณ์ไฟฟ้า'),
  ('วัสดุสิ้นเปลือง'),
  ('อื่นๆ');
