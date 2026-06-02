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
  image_key    TEXT,              -- R2 object key (preferred); image served via /api/items/:id/image
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

-- seed categories
INSERT OR IGNORE INTO categories (name) VALUES
  ('ชิ้นส่วนเครื่องจักร'),
  ('เครื่องมือช่าง'),
  ('อุปกรณ์ไฟฟ้า'),
  ('วัสดุสิ้นเปลือง'),
  ('อื่นๆ');
