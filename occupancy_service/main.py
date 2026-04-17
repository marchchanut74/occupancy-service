"""
main.py
───────
Occupancy Detection Service — Entry Point

รวมทุกอย่าง:
  - โหลด config
  - สร้าง preprocessors ต่อกล้อง
  - โหลด YOLO model
  - scheduler ดึงภาพ + detect ตาม interval
  - FastAPI server ให้ระบบภายนอกเรียก

Usage:
    python -m occupancy_service.main
    python -m occupancy_service.main --config config.yaml
"""

import cv2
import sys
import yaml
import time
import logging
import argparse
import threading
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from .preprocessing import Preprocessor
from .inference import Detector
from . import store
from .api import router as api_router, init_api
from .mqtt_publisher import MQTTPublisher, HAS_MQTT

# ─────────────────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────────────────
logger = logging.getLogger("occupancy")
_scheduler_stop = threading.Event()

# Cleanup schedule — รันวันละครั้งไม่ใช่ทุก cycle
CLEANUP_INTERVAL_SEC = 86400  # 24 ชม.
_last_cleanup = 0.0

# File read retry (กันไฟล์ยังเขียนไม่เสร็จ)
FILE_READ_RETRIES = 2
FILE_READ_RETRY_DELAY_SEC = 0.5


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(cfg: dict):
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("file")

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


# ─────────────────────────────────────────────────────────
# Robust file read — retry กันภาพยังเขียนไม่เสร็จ
# ─────────────────────────────────────────────────────────
def read_frame_safe(path: str, cam_id: str):
    """
    อ่านภาพด้วย retry — กัน race condition กับตัว admin ที่เขียนไฟล์
    - ถ้า imread return None → รอสักนิดแล้วลองใหม่
    - ถ้ายังไม่ได้ → return None (caller จะ skip)
    """
    for attempt in range(FILE_READ_RETRIES + 1):
        frame = cv2.imread(path)
        if frame is not None:
            return frame
        if attempt < FILE_READ_RETRIES:
            logger.debug(f"[{cam_id}] Read failed (attempt {attempt+1}), "
                         f"retrying in {FILE_READ_RETRY_DELAY_SEC}s")
            time.sleep(FILE_READ_RETRY_DELAY_SEC)
    return None


# ─────────────────────────────────────────────────────────
# Watch-folder scheduler
# ─────────────────────────────────────────────────────────
def run_detection_cycle(cameras: list[dict], preprocessors: dict,
                        detector: Detector, mqtt_pub=None):
    """detect 1 รอบ — ดึงภาพทุกกล้อง → preprocess → inference → save"""
    for cam in cameras:
        cam_id = cam["id"]
        input_path = cam.get("input_path", "")

        if not input_path or not Path(input_path).exists():
            logger.debug(f"[{cam_id}] No image at {input_path}")
            continue

        # อ่านภาพ (with retry)
        frame = read_frame_safe(input_path, cam_id)
        if frame is None:
            logger.warning(f"[{cam_id}] Cannot read: {input_path}")
            continue

        # preprocess
        pp = preprocessors.get(cam_id)
        if pp:
            frame = pp.process(frame)

        # detect
        result = detector.detect(frame, camera_id=cam_id)

        # save
        store.save_detection(result)

        # MQTT
        if mqtt_pub:
            mqtt_pub.publish(result)

        logger.info(
            f"[{cam_id}] persons={result.person_count}  "
            f"conf={result.max_confidence:.2f}  "
            f"{result.inference_ms:.0f}ms"
        )


def maybe_cleanup():
    """รัน cleanup วันละครั้ง — จำเวลาที่รันล่าสุดไว้"""
    global _last_cleanup
    now = time.time()
    if now - _last_cleanup >= CLEANUP_INTERVAL_SEC:
        try:
            store.cleanup_old(days=30)
            _last_cleanup = now
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


def scheduler_loop(cfg: dict, preprocessors: dict,
                   detector: Detector, mqtt_pub=None):
    """Background thread: detect ตาม interval"""
    cameras = cfg.get("cameras", [])
    interval = cfg.get("schedule", {}).get("interval_sec", 300)
    logger.info(f"Scheduler started — interval={interval}s  "
                f"cameras={len(cameras)}")

    while not _scheduler_stop.is_set():
        try:
            run_detection_cycle(cameras, preprocessors, detector, mqtt_pub)
        except Exception as e:
            logger.error(f"Detection cycle error: {e}", exc_info=True)

        # cleanup เช็คทุก cycle แต่รันจริงวันละครั้ง
        maybe_cleanup()

        _scheduler_stop.wait(timeout=interval)

    logger.info("Scheduler stopped")


# ─────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────
def create_app(cfg: dict) -> FastAPI:
    """สร้าง FastAPI app + เริ่ม scheduler"""

    # ── สร้าง preprocessors ──
    preprocessors = {}
    for cam in cfg.get("cameras", []):
        cam_id = cam["id"]
        preprocessors[cam_id] = Preprocessor(
            camera_id=cam_id,
            mask_path=cam.get("mask_path"),
            resize_width=cam.get("resize_width", 640),
        )
    logger.info(f"Preprocessors: {list(preprocessors.keys())}")

    # ── โหลด model ──
    model_cfg = cfg.get("model", {})
    detector = Detector(
        model_path=model_cfg.get("path", "yolo26s.pt"),
        confidence=model_cfg.get("confidence", 0.1),
        person_class=model_cfg.get("person_class", 0),
    )

    # ── MQTT (optional) ──
    mqtt_pub = None
    mqtt_cfg = cfg.get("mqtt", {})
    if mqtt_cfg.get("enabled") and HAS_MQTT:
        try:
            mqtt_pub = MQTTPublisher(
                broker=mqtt_cfg["broker"],
                port=mqtt_cfg.get("port", 1883),
                topic_prefix=mqtt_cfg.get("topic_prefix", "occupancy"),
                username=mqtt_cfg.get("username"),
                password=mqtt_cfg.get("password"),
            )
        except Exception as e:
            logger.error(f"MQTT init failed: {e}")

    # ── Init DB ──
    store.init_db()

    # ── Init API ──
    init_api(detector, preprocessors, mqtt_pub)

    # ── Lifespan ──
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # startup: เริ่ม scheduler thread
        t = threading.Thread(
            target=scheduler_loop,
            args=(cfg, preprocessors, detector, mqtt_pub),
            daemon=True,
        )
        t.start()
        logger.info("Service started")
        yield
        # shutdown
        _scheduler_stop.set()
        if mqtt_pub:
            mqtt_pub.stop()
        logger.info("Service stopped")

    app = FastAPI(
        title="Occupancy Detection Service",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(api_router)

    return app


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Occupancy Detection Service")
    parser.add_argument("--config", default="config.yaml",
                        help="path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    api_cfg = cfg.get("api", {})
    host = api_cfg.get("host", "0.0.0.0")
    port = api_cfg.get("port", 8000)

    app = create_app(cfg)

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
