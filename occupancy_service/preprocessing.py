"""
preprocessing.py
────────────────
Preprocessing pipeline: resize → mask → ready for inference

ลำดับ resize ก่อน mask เพราะ:
  - ไม่รู้ว่า input จะเข้ามาขนาดเท่าไหร่ (แต่ละกล้องอาจต่างกัน)
  - resize ก่อนให้ได้ขนาดคงที่ (เช่น 640xH)
  - mask file จึงทำแค่ขนาดเดียว ตรงกับภาพหลัง resize
  - ไม่ต้อง resize mask ทุกครั้ง → เร็วกว่า แม่นกว่า

Mask logic:
  - mask image เป็นไฟล์ขาว-ดำ ขนาดตรงกับภาพหลัง resize
  - white = detect (ในห้อง), black = ignore (นอกห้อง/กระจก)
  - apply mask แล้ว pixel ที่ mask=ดำ จะโดนระบายดำบนภาพจริง
  - ภาพส่วนที่เหลือยังเป็นสีปกติ YOLO detect ได้ตามปกติ
"""

import cv2
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Preprocessing pipeline ต่อ 1 กล้อง
    ลำดับ: resize → mask
    """

    def __init__(self, camera_id: str,
                 mask_path: str | None = None,
                 resize_width: int = 640):
        self.camera_id = camera_id
        self.resize_width = resize_width
        self.mask = None

        if mask_path and Path(mask_path).exists():
            raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if raw is not None:
                # threshold ให้เป็น binary (0 หรือ 255)
                _, self.mask = cv2.threshold(raw, 127, 255, cv2.THRESH_BINARY)
                logger.info(f"[{camera_id}] Loaded mask: {mask_path}  "
                            f"shape={self.mask.shape}")
            else:
                logger.warning(f"[{camera_id}] Cannot read mask: {mask_path}")
        else:
            if mask_path:
                logger.warning(f"[{camera_id}] Mask not found: {mask_path}")

    def resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize ภาพให้ความกว้าง = resize_width (รักษา aspect ratio)"""
        if self.resize_width <= 0:
            return frame

        h, w = frame.shape[:2]
        if w <= self.resize_width:
            return frame

        scale = self.resize_width / w
        new_w = self.resize_width
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def apply_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply mask บนภาพ (หลัง resize แล้ว)
        pixel ที่ mask=0 (ดำ) → ภาพจริงตรงนั้นเป็นดำ
        ภาพที่ออกมายังเป็นสี RGB ปกติ ไม่ใช่ขาวดำ
        """
        if self.mask is None:
            return frame

        h, w = frame.shape[:2]
        mask = self.mask

        # ตรวจสอบขนาด — ปกติควรตรงกันแล้วหลัง resize
        # แต่ถ้าไม่ตรง (เช่น aspect ratio ต่างกัน) ให้ resize mask ให้ตรง
        if mask.shape[:2] != (h, w):
            logger.warning(
                f"[{self.camera_id}] Mask size {mask.shape[:2]} "
                f"!= frame size {(h, w)} — resizing mask. "
                f"ควรสร้าง mask ใหม่ให้ตรงกับภาพหลัง resize"
            )
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # สร้าง 3-channel mask
        mask_3ch = cv2.merge([mask, mask, mask])

        # apply: pixel ที่ mask=0 → ภาพเป็นดำ, ที่เหลือเหมือนเดิม
        masked = cv2.bitwise_and(frame, mask_3ch)
        return masked

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Full pipeline:  resize → mask
        1. resize ก่อน — ทำให้ขนาดคงที่ ไม่ว่า input จะมาขนาดไหน
        2. mask ทีหลัง — mask file ทำขนาดเดียว ตรงกับภาพหลัง resize
        input: BGR frame จากกล้อง (ขนาดอะไรก็ได้)
        output: BGR frame พร้อม inference (ขนาด resize_width x H)
        """
        out = self.resize(frame)
        out = self.apply_mask(out)
        return out
