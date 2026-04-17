"""
inference.py
────────────
YOLO inference wrapper — โหลดโมเดลครั้งเดียว แล้วใช้ detect ซ้ำได้เรื่อยๆ
"""

import time
import logging
import numpy as np
from dataclasses import dataclass, field
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """ผลลัพธ์จากการ detect 1 ภาพ"""
    camera_id: str
    occupied: bool
    person_count: int
    max_confidence: float
    inference_ms: float
    timestamp: str               # ISO format
    boxes: list = field(default_factory=list)   # list of [x1,y1,x2,y2]
    confidences: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "occupied": self.occupied,
            "person_count": self.person_count,
            "max_confidence": round(self.max_confidence, 3),
            "inference_ms": round(self.inference_ms, 1),
            "timestamp": self.timestamp,
        }


class Detector:
    """YOLO person detector"""

    def __init__(self, model_path: str, confidence: float = 0.1,
                 person_class: int = 0):
        logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.person_class = person_class
        logger.info(f"Model loaded — conf={confidence}")

    def detect(self, frame: np.ndarray, camera_id: str = "") -> DetectionResult:
        """
        Detect persons ใน 1 frame
        input:  BGR frame (preprocessed แล้ว)
        output: DetectionResult
        """
        from datetime import datetime, timezone

        t0 = time.perf_counter()

        results = self.model(
            frame,
            conf=self.confidence,
            classes=[self.person_class],
            verbose=False,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        boxes = []
        confs = []
        if results and results[0].boxes and len(results[0].boxes):
            boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
            confs = results[0].boxes.conf.cpu().numpy().tolist()

        n = len(boxes)
        max_conf = float(max(confs)) if confs else 0.0

        now = datetime.now(timezone.utc).isoformat()

        return DetectionResult(
            camera_id=camera_id,
            occupied=(n > 0),
            person_count=n,
            max_confidence=max_conf,
            inference_ms=elapsed_ms,
            timestamp=now,
            boxes=boxes,
            confidences=confs,
        )
