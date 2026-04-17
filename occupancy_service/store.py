"""
store.py
────────
SQLite store — เก็บผลลัพธ์การ detect ไว้สำหรับ API query ย้อนหลัง

WAL mode (Write-Ahead Logging):
  - เปิดไว้เพื่อให้ reader/writer ทำงานพร้อมกันได้ดีขึ้น
  - ลดปัญหา "database is locked" ในกรณีที่ scheduler กับ API ทำงานพร้อมกัน
  - เปิดครั้งเดียวตอน init_db() → persist ตลอดอายุไฟล์ DB
"""

import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = "data/occupancy.db"


def _ensure_dir():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)


def init_db():
    """สร้างตาราง (ถ้ายังไม่มี) + เปิด WAL mode"""
    _ensure_dir()
    with _connect() as conn:
        # WAL mode: reader ไม่บล็อก writer, writer ไม่บล็อก reader
        conn.execute("PRAGMA journal_mode=WAL;")
        # synchronous=NORMAL + WAL = เร็วพอ ปลอดภัยพอสำหรับ logging use case นี้
        conn.execute("PRAGMA synchronous=NORMAL;")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id   TEXT NOT NULL,
                occupied    INTEGER NOT NULL,
                person_count INTEGER NOT NULL,
                max_confidence REAL,
                inference_ms   REAL,
                timestamp   TEXT NOT NULL,
                created_at  TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_camera_ts
            ON detections (camera_id, timestamp)
        """)
    logger.info(f"Database ready (WAL mode): {DB_PATH}")


@contextmanager
def _connect():
    # timeout=5.0 → รอ 5 วินาทีถ้ามีใครล็อกอยู่ ก่อนจะ raise
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_detection(result) -> int:
    """บันทึกผลลัพธ์ 1 record"""
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute("""
            INSERT INTO detections
                (camera_id, occupied, person_count, max_confidence,
                 inference_ms, timestamp, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            result.camera_id,
            int(result.occupied),
            result.person_count,
            result.max_confidence,
            result.inference_ms,
            result.timestamp,
            now,
        ))
        return cur.lastrowid


def get_latest(camera_id: str) -> dict | None:
    """ดึงผลล่าสุดของกล้อง"""
    with _connect() as conn:
        row = conn.execute("""
            SELECT * FROM detections
            WHERE camera_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (camera_id,)).fetchone()
        return dict(row) if row else None


def get_latest_all() -> list[dict]:
    """ดึงผลล่าสุดของทุกกล้อง"""
    with _connect() as conn:
        rows = conn.execute("""
            SELECT d.* FROM detections d
            INNER JOIN (
                SELECT camera_id, MAX(timestamp) AS max_ts
                FROM detections
                GROUP BY camera_id
            ) latest ON d.camera_id = latest.camera_id
                    AND d.timestamp = latest.max_ts
            ORDER BY d.camera_id
        """).fetchall()
        return [dict(r) for r in rows]


def get_history(camera_id: str, hours: int = 24) -> list[dict]:
    """ดึง history ย้อนหลัง n ชั่วโมง"""
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with _connect() as conn:
        rows = conn.execute("""
            SELECT * FROM detections
            WHERE camera_id = ? AND timestamp >= ?
            ORDER BY timestamp ASC
        """, (camera_id, cutoff)).fetchall()
        return [dict(r) for r in rows]


def cleanup_old(days: int = 30):
    """ลบข้อมูลเก่ากว่า n วัน"""
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with _connect() as conn:
        cur = conn.execute("""
            DELETE FROM detections WHERE timestamp < ?
        """, (cutoff,))
        if cur.rowcount > 0:
            logger.info(f"Cleaned up {cur.rowcount} old records")
