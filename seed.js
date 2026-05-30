/**
 * Seed real sample data into the live D1 database via the Worker API.
 * Builds small valid PNG thumbnails (so embeddings get created) and POSTs items.
 * Run:  node seed.js
 */
const https = require('https');
const zlib = require('zlib');

const BASE = 'https://requisition-system.wiphawas-sketchup.workers.dev';
const ADMIN = 'admin1234';

// ---- minimal PNG encoder (solid colour, 96x96) ----
const CRC_TABLE = (() => {
  const t = [];
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length, 0);
  const typeBuf = Buffer.from(type, 'ascii');
  const crcBuf = Buffer.alloc(4);
  crcBuf.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0);
  return Buffer.concat([len, typeBuf, data, crcBuf]);
}
function solidPng(r, g, b, size = 96) {
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0); ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0; // 8-bit RGB
  const rowLen = 1 + size * 3;
  const raw = Buffer.alloc(rowLen * size);
  for (let y = 0; y < size; y++) {
    const off = y * rowLen; raw[off] = 0; // filter none
    for (let x = 0; x < size; x++) {
      const p = off + 1 + x * 3;
      // simple diagonal tint so it's not a pure flat colour
      const t = ((x + y) % 32) < 16 ? 0 : 24;
      raw[p] = Math.min(255, r + t); raw[p + 1] = Math.min(255, g + t); raw[p + 2] = Math.min(255, b + t);
    }
  }
  const idat = zlib.deflateSync(raw);
  const png = Buffer.concat([sig, chunk('IHDR', ihdr), chunk('IDAT', idat), chunk('IEND', Buffer.alloc(0))]);
  return 'data:image/png;base64,' + png.toString('base64');
}

function post(path, body) {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify(body);
    const u = new URL(BASE + path);
    const req = https.request({
      hostname: u.hostname, path: u.pathname + u.search, method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload), 'X-Admin-Key': ADMIN },
    }, res => {
      let d = ''; res.on('data', c => d += c);
      res.on('end', () => { try { resolve(JSON.parse(d)); } catch { resolve({ raw: d, status: res.statusCode }); } });
    });
    req.on('error', reject); req.write(payload); req.end();
  });
}

// cat ids (from live DB): 1=ชิ้นส่วนเครื่องจักร 2=เครื่องมือช่าง 3=อุปกรณ์ไฟฟ้า 4=วัสดุสิ้นเปลือง 5=อื่นๆ
const ITEMS = [
  { part_code: 'BRG-6204', name: 'ตลับลูกปืน 6204 ZZ', category_id: 1, unit: 'ตลับ', stock_qty: 24, min_stock: 6, desc: 'ตลับลูกปืนเม็ดกลม 6204 ฝาเหล็ก 2 ข้าง', rgb: [70, 90, 120] },
  { part_code: 'BLT-V40', name: 'สายพานร่อง V เบอร์ A40', category_id: 1, unit: 'เส้น', stock_qty: 5, min_stock: 8, desc: 'สายพานส่งกำลังร่อง V ขนาด A40', rgb: [40, 40, 45] },
  { part_code: 'WR-17', name: 'ประแจแหวน 17 มม.', category_id: 2, unit: 'อัน', stock_qty: 0, min_stock: 3, desc: 'ประแจแหวนข้างปากตาย 17mm เหล็ก Cr-V', rgb: [120, 125, 130] },
  { part_code: 'DRL-10', name: 'ดอกสว่านเหล็ก HSS 10 มม.', category_id: 2, unit: 'ดอก', stock_qty: 18, min_stock: 5, desc: 'ดอกสว่านเจาะเหล็ก HSS ขนาด 10mm', rgb: [150, 140, 90] },
  { part_code: 'MCB-C16', name: 'เบรกเกอร์ลูกย่อย 16A 1P', category_id: 3, unit: 'ตัว', stock_qty: 30, min_stock: 10, desc: 'MCB 1 Pole 16A curve C', rgb: [180, 170, 60] },
  { part_code: 'CBL-THW25', name: 'สายไฟ THW 2.5 ตร.มม. (ดำ)', category_id: 3, unit: 'เมตร', stock_qty: 4, min_stock: 20, desc: 'สายไฟทองแดงหุ้ม THW 2.5 sq.mm สีดำ', rgb: [30, 30, 30] },
  { part_code: 'GLV-CT', name: 'ถุงมือผ้าทอ (โหล)', category_id: 4, unit: 'โหล', stock_qty: 2, min_stock: 10, desc: 'ถุงมือผ้าทอ 7 ขีด สำหรับงานทั่วไป', rgb: [210, 200, 180] },
  { part_code: 'OIL-46', name: 'น้ำมันไฮดรอลิก ISO 46 (18L)', category_id: 4, unit: 'ถัง', stock_qty: 7, min_stock: 3, desc: 'น้ำมันไฮดรอลิกเบอร์ 46 ถัง 18 ลิตร', rgb: [150, 110, 40] },
  { part_code: 'TAPE-PVC', name: 'เทปพันสายไฟ PVC สีดำ', category_id: 5, unit: 'ม้วน', stock_qty: 40, min_stock: 15, desc: 'เทปพันสายไฟ PVC 0.18mm x 10m', rgb: [25, 25, 25] },
  { part_code: 'SCR-M8X20', name: 'สกรูหัวหกเหลี่ยม M8x20 (กล่อง)', category_id: 5, unit: 'กล่อง', stock_qty: 12, min_stock: 4, desc: 'สกรู Hex bolt M8x20 ชุบซิงค์ กล่อง 100 ตัว', rgb: [160, 165, 170] },
];

(async () => {
  console.log('Seeding', ITEMS.length, 'items...');
  for (const it of ITEMS) {
    const res = await post('/api/items', {
      part_code: it.part_code, name: it.name, description: it.desc,
      category_id: it.category_id, unit: it.unit,
      stock_qty: it.stock_qty, min_stock: it.min_stock,
      image_b64: solidPng(it.rgb[0], it.rgb[1], it.rgb[2]),
    });
    console.log(res.ok ? `  OK  ${it.part_code}  (id ${res.id})` : `  SKIP ${it.part_code}: ${res.error || JSON.stringify(res)}`);
  }
  console.log('Done.');
})();
