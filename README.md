# Occupancy Detection Service

ระบบตรวจจับคนในห้อง (Occupancy Detection) โดยใช้ YOLO
รับภาพจากกล้อง → preprocess (resize + mask) → inference → ส่งผลผ่าน REST API / MQTT

---

## Requirements

- **Python 3.10 ขึ้นไป** (ใช้ union type syntax `str | None`)
- GPU (optional — ถ้ามีจะเร็วกว่า CPU หลายเท่า)
- Disk space อย่างน้อย 2GB (สำหรับโมเดล + SQLite history)

---

## Architecture

```
Admin วางภาพ / POST ภาพ
        │
        ▼
┌──────────────────────┐
│  Preprocessing       │
│  1. resize to 640    │
│  2. apply mask       │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  YOLO Inference      │
│  yolo26s.pt          │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Result Store        │
│  (SQLite)            │
└──────────┬───────────┘
           │
     ┌─────┼──────┐
     ▼     ▼      ▼
  REST   MQTT   Logs
  API    (HA)
```

**ลำดับ preprocessing = resize ก่อน mask** เพราะ:
- ภาพจากกล้องอาจมาขนาดไหนก็ได้ (1080p, 4K, ฯลฯ)
- resize ให้เป็น 640 ก่อน → ทุกภาพมีขนาดคงที่
- mask file ทำครั้งเดียวที่ขนาด 640 → ใช้ได้ตลอด

---

## Folder Structure

```
occupancy_service/
├── config.yaml                  ← config ทุกอย่าง
├── requirements.txt
├── Dockerfile
├── yolo26s.pt                   ← โมเดล YOLO (วางเอง)
│
├── masks/                       ← mask ขาว-ดำ (วาดเอง)
│   ├── camera_01_mask.png
│   └── camera_02_mask.png
│
├── occupancy_service/           ← source code
│   ├── __init__.py
│   ├── __main__.py
│   ├── main.py                  ← entry point
│   ├── preprocessing.py         ← resize → mask
│   ├── inference.py             ← YOLO detect
│   ├── store.py                 ← SQLite
│   ├── api.py                   ← REST API
│   └── mqtt_publisher.py        ← MQTT (optional)
│
├── config_occupancy_ha.yaml     ← วางใน HA config
│
├── data/                        ← สร้างเองตอนรัน
│   └── occupancy.db
│
└── logs/                        ← สร้างเองตอนรัน
    └── occupancy.log
```

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. วาง YOLO model

วาง `yolo26s.pt` ไว้ใน directory เดียวกับ config.yaml

### 3. สร้าง mask (ถ้าต้องใช้)

ใช้ `mask_editor.html` วาด mask สำหรับกล้องที่มีกระจก
(โหลดภาพที่ resize 640 แล้วไปวาด เพื่อให้ mask ขนาดตรงกัน)
แล้ววาง mask PNG ไว้ใน `masks/`

### 4. แก้ config.yaml

```yaml
cameras:
  - id: "camera_01"
    input_path: "/data/snapshots/camera_01/latest.jpg"
    mask_path: "masks/camera_01_mask.png"
    resize_width: 640
```

### 5. รัน

```bash
python -m occupancy_service.main --config config.yaml
```

Server จะเริ่มที่ `http://0.0.0.0:8000`

---

## API Endpoints

### GET /api/health
ตรวจสอบสถานะระบบ

### GET /api/status
สถานะล่าสุดของทุกกล้อง

```json
{
  "cameras": [
    {
      "camera_id": "camera_01",
      "occupied": true,
      "person_count": 2,
      "max_confidence": 0.87,
      "inference_ms": 45.2,
      "timestamp": "2026-04-12T14:30:00+00:00"
    }
  ]
}
```

### GET /api/status/{camera_id}
สถานะล่าสุดของกล้องเดียว

### GET /api/history/{camera_id}?hours=24
ดึง history ย้อนหลัง (default 24 ชม., max 168 ชม./7 วัน)

### POST /api/detect/{camera_id}
Upload ภาพมา detect ทันที

```bash
curl -X POST http://localhost:8000/api/detect/camera_01 \
  -F "file=@snapshot.jpg"
```

---

## Input Modes — รองรับ 2 แบบ

### 1. Watch Folder Mode
ระบบจะอ่านภาพจาก `input_path` ที่กำหนดใน config ทุก `interval_sec` วินาที
Admin แค่วางภาพล่าสุดลงใน folder ที่กำหนด (เขียนทับไฟล์เดิมได้)

### 2. POST API Mode
Admin POST ภาพตรงๆ มาที่ `/api/detect/{camera_id}` ได้ทันที
ไม่ต้องรอ scheduler

ทั้งสองแบบทำงานพร้อมกันได้

---

## Home Assistant Integration

วาง `config_occupancy_ha.yaml` ไว้ใน HA config directory
แล้ว include ใน configuration.yaml:

```yaml
homeassistant:
  packages:
    occupancy: !include config_occupancy_ha.yaml
```

HA จะ poll API ทุก 5 นาที แล้วสร้าง sensor ให้แต่ละกล้อง

---

## Docker

```bash
docker build -t occupancy-service .
docker run -d \
  -p 8000:8000 \
  -v /path/to/snapshots:/data/snapshots \
  -v /path/to/yolo26s.pt:/app/yolo26s.pt \
  -v /path/to/masks:/app/masks \
  occupancy-service
```

---

## Mask คืออะไร?

Mask เป็นไฟล์ภาพขาว-ดำ **แยกจากภาพกล้อง** ที่บอกว่า:
- **ขาว** = บริเวณที่ต้องการ detect (ในห้อง)
- **ดำ** = บริเวณที่ไม่ต้องการ detect (นอกห้อง, ผ่านกระจก)

ก่อน inference ระบบจะเอา mask มา **ระบายทับ** เฉพาะบริเวณดำ
บนภาพต้นฉบับ → pixel ตรงนั้นเป็นดำ → YOLO ไม่เห็นคน
**ภาพส่วนที่เหลือยังเป็นสีปกติ** YOLO detect ได้ตามปกติ

---

## Troubleshooting

### `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`
ใช้ Python เวอร์ชันต่ำกว่า 3.10 — อัปเดตเป็น 3.10+

### Detect ไม่เจอคนเลย ถึงจะมีคนในภาพ
ตรวจสอบ mask file:
- เปิดดู mask PNG — ควรเห็นบริเวณในห้องเป็นสีขาว, นอกห้องเป็นสีดำ
- ถ้าเห็นเป็นสีดำทั้งภาพ → วาด mask ผิดข้าง ให้วาดใหม่

### Detect เจอคนนอกห้อง (ผ่านกระจก)
ขยาย mask zone ให้ครอบคลุมบริเวณกระจกมากขึ้น

---

## Performance & Reliability Notes

- **Event loop ไม่บล็อก** — POST `/api/detect` ใช้ sync def ซึ่ง FastAPI จะโยนไป
  threadpool ให้เอง ระหว่างที่ YOLO รันอยู่ GET requests อื่นๆ ยังตอบได้
- **SQLite WAL mode** — เปิดไว้ในตัว ลดปัญหา "database is locked" เมื่อ scheduler
  กับ API เขียนพร้อมกัน
- **Cleanup วันละครั้ง** — ลบข้อมูลเก่ากว่า 30 วันอัตโนมัติ ไม่รัน DELETE ทุก 5 นาที
- **File read retry** — ถ้า admin ยังเขียน `latest.jpg` ไม่เสร็จตอนที่ scheduler
  มาอ่าน ระบบจะ retry 1 ครั้ง รอ 0.5 วินาที ก่อนจะ skip

