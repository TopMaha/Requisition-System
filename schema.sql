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
  image_b64    TEXT,              -- base64 image
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

-- seed categories
INSERT OR IGNORE INTO categories (name) VALUES
  ('ชิ้นส่วนเครื่องจักร'),
  ('เครื่องมือช่าง'),
  ('อุปกรณ์ไฟฟ้า'),
  ('วัสดุสิ้นเปลือง'),
  ('อื่นๆ');
