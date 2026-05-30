# ระบบเบิกอุปกรณ์/อะไหล่โรงงาน
## Factory Requisition System — Image AI Matching

---

## ไฟล์ในโปรเจค

| ไฟล์ | คำอธิบาย |
|------|----------|
| `index.html` | หน้าเว็บหลัก (เปิดในเบราว์เซอร์ได้เลย) |
| `worker.js` | Cloudflare Worker API + AI Logic |
| `wrangler.toml` | Config สำหรับ Cloudflare |
| `schema.sql` | สร้าง Tables ใน D1 Database |
| `deploy.bat` | Script สำหรับ Deploy (รันครั้งเดียว) |

---

## วิธีติดตั้ง (ครั้งแรก)

### ขั้นที่ 1 — ติดตั้ง Node.js และ Wrangler
```
npm install -g wrangler
```

### ขั้นที่ 2 — รัน deploy.bat
ดับเบิลคลิก `deploy.bat` ทำตามขั้นตอน:
1. Login Cloudflare
2. สร้าง D1 database → copy `database_id`
3. วาง `database_id` ใน `wrangler.toml`
4. Deploy Worker

### ขั้นที่ 3 — ตั้งค่าใน index.html
1. เปิด `index.html` ในเบราว์เซอร์
2. กดปุ่ม ⚙️ มุมบนขวา
3. ใส่ Worker URL (เช่น `https://requisition-system.xxx.workers.dev`)
4. ใส่ Admin Password (default: `admin1234`)
5. กดบันทึก

---

## วิธีใช้งาน

### สำหรับพนักงาน (เบิกของ)
1. เปิด `index.html`
2. กดแท็บ **📷 สแกน/เบิก**
3. ถ่ายรูปหรืออัปโหลดรูปอุปกรณ์ที่ต้องการเบิก
4. กด **"ค้นหาอุปกรณ์ด้วย AI"**
5. เลือกอุปกรณ์ที่ตรงกันจากผลลัพธ์
6. กรอกชื่อและจำนวน → กด **"ส่งใบเบิก"**

### สำหรับ Admin (จัดการระบบ)
1. กดแท็บ **🔧 Admin**
2. ใส่รหัสผ่าน (default: `admin1234`)
3. เพิ่มอุปกรณ์ใหม่พร้อมรูปภาพ
4. อนุมัติ/ปฏิเสธใบเบิก
5. ปรับ Stock

---

## ระบบ AI Image Matching

ใช้ **Cloudflare Workers AI** — model `@cf/openai/clip-vit-base-patch32`
- เมื่อ Admin เพิ่มอุปกรณ์พร้อมรูป → ระบบสร้าง CLIP embedding อัตโนมัติ
- เมื่อ User ถ่ายรูป → ระบบเปรียบเทียบ cosine similarity กับ embeddings ทั้งหมดในฐานข้อมูล
- แสดงผลพร้อม % ความมั่นใจ (top 5 รายการ)

**Free tier:** 10,000 AI requests/วัน (Workers AI Free)

---

## Database Schema

- `categories` — หมวดหมู่อุปกรณ์
- `items` — รายการอุปกรณ์/อะไหล่ (มี embedding)
- `requisitions` — ใบเบิก
- `requisition_items` — รายการในใบเบิก
- `stock_movements` — ประวัติการเคลื่อนไหว Stock

---

## เปลี่ยน Admin Password

แก้ในไฟล์ `wrangler.toml`:
```toml
[vars]
ADMIN_PASSWORD = "รหัสใหม่"
```
แล้วรัน `wrangler deploy` อีกครั้ง

---

*สร้างโดย Calue AI — 2026*
