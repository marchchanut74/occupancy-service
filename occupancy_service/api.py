"""
api.py
──────
REST API endpoints — ให้ระบบภายนอกเรียกดูผล detection

Note on async vs sync:
  - GET endpoints ที่อ่าน SQLite → sync def (เร็วพอ ไม่จำเป็นต้อง async)
  - POST /detect → sync def เพราะต้องรัน YOLO (CPU bound)
    ถ้าใช้ async def จะบล็อก event loop ทั้งตัว
    FastAPI จะโยน sync def ไปรันใน threadpool ให้เอง → ไม่บล็อก event loop
"""

import cv2
import numpy as np
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from . import store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["occupancy"])

# references ที่ main.py จะ inject เข้ามา
_detector = None
_preprocessors = {}
_mqtt = None


def init_api(detector, preprocessors, mqtt_pub=None):
    """เรียกจาก main.py ตอน startup"""
    global _detector, _preprocessors, _mqtt
    _detector = detector
    _preprocessors = preprocessors
    _mqtt = mqtt_pub


# ── GET /api/status ──────────────────────────────────────
@router.get("/status")
def get_status_all():
    """สถานะล่าสุดของทุกกล้อง"""
    rows = store.get_latest_all()
    if not rows:
        return {"cameras": [], "message": "No detection data yet"}
    return {"cameras": rows}


# ── GET /api/status/{camera_id} ──────────────────────────
@router.get("/status/{camera_id}")
def get_status(camera_id: str):
    """สถานะล่าสุดของกล้องเดียว"""
    row = store.get_latest(camera_id)
    if not row:
        raise HTTPException(404, f"No data for camera: {camera_id}")
    return row


# ── GET /api/history/{camera_id} ─────────────────────────
@router.get("/history/{camera_id}")
def get_history(camera_id: str, hours: int = Query(24, ge=1, le=168)):
    """ดึง history ย้อนหลัง (default 24 ชม., max 7 วัน)"""
    rows = store.get_history(camera_id, hours)
    return {
        "camera_id": camera_id,
        "hours": hours,
        "count": len(rows),
        "detections": rows,
    }


# ── POST /api/detect/{camera_id} ────────────────────────
# sync def — FastAPI จะโยนไป threadpool ให้เอง ไม่บล็อก event loop
@router.post("/detect/{camera_id}")
def detect_upload(camera_id: str, file: UploadFile = File(...)):
    """
    Admin POST ภาพมา detect ทันที
    - ระบบหยิบ preprocessor ของกล้องนั้น (resize + mask)
    - รัน inference
    - บันทึกผล + publish MQTT
    - return ผลลัพธ์
    """
    if _detector is None:
        raise HTTPException(503, "Detector not initialized")

    # อ่านภาพจาก upload (sync — ไม่มี await)
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(400, "Cannot decode image")

    # preprocess (ใช้ config ของกล้องนั้น ถ้ามี)
    preprocessor = _preprocessors.get(camera_id)
    if preprocessor:
        frame = preprocessor.process(frame)
    else:
        logger.warning(f"No preprocessor for camera {camera_id}, "
                       f"using raw image")

    # detect (CPU-heavy — รันใน threadpool)
    result = _detector.detect(frame, camera_id=camera_id)

    # save to DB
    store.save_detection(result)

    # publish MQTT
    if _mqtt:
        _mqtt.publish(result)

    logger.info(
        f"[{camera_id}] POST detect: "
        f"persons={result.person_count}  "
        f"conf={result.max_confidence:.2f}  "
        f"{result.inference_ms:.0f}ms"
    )

    return result.to_dict()


# ── GET /api/health ──────────────────────────────────────
@router.get("/health")
def health():
    return {
        "status": "ok",
        "detector_loaded": _detector is not None,
        "cameras": list(_preprocessors.keys()),
        "time": datetime.now(timezone.utc).isoformat(),
    }
