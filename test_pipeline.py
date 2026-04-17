"""
test_pipeline.py
────────────────
Standalone test — ทดสอบ preprocessing (resize + mask) + YOLO detect
โดยไม่ต้อง setup FastAPI / SQLite / MQTT

Usage:
    # ทดสอบภาพเดียว
    python test_pipeline.py --image sample.jpg --mask masks/camera_01_mask.png

    # ทดสอบโดยไม่ใช้ mask
    python test_pipeline.py --image sample.jpg

    # ทดสอบหลายภาพใน folder
    python test_pipeline.py --folder ./test_images --mask masks/camera_01_mask.png

    # ปรับ confidence / resize
    python test_pipeline.py --image sample.jpg --conf 0.25 --resize 640

Output:
    - แสดงผลใน terminal: จำนวนคน, confidence, เวลา inference
    - บันทึกภาพผล (มี bounding box + mask overlay) ไว้ที่ ./test_output/
"""

import cv2
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Optional
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────
# Preprocessing (copy logic จาก preprocessing.py)
# ─────────────────────────────────────────────────────────
class Preprocessor:
    def __init__(self, mask_path: Optional[str] = None, resize_width: int = 640):
        self.resize_width = resize_width
        self.mask = None

        if mask_path and Path(mask_path).exists():
            raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if raw is not None:
                _, self.mask = cv2.threshold(raw, 127, 255, cv2.THRESH_BINARY)
                print(f"[Mask] Loaded: {mask_path}  shape={self.mask.shape}")
            else:
                print(f"[Mask] Cannot read: {mask_path}")
        elif mask_path:
            print(f"[Mask] Not found: {mask_path}")
        else:
            print(f"[Mask] No mask (full image)")

    def resize(self, frame):
        h, w = frame.shape[:2]
        if self.resize_width <= 0 or w <= self.resize_width:
            return frame
        scale = self.resize_width / w
        return cv2.resize(frame, (self.resize_width, int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    def apply_mask(self, frame):
        if self.mask is None:
            return frame
        h, w = frame.shape[:2]
        mask = self.mask
        if mask.shape[:2] != (h, w):
            print(f"  [warn] mask size {mask.shape[:2]} != frame {(h,w)}, "
                  f"auto-resizing mask")
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_3ch = cv2.merge([mask, mask, mask])
        return cv2.bitwise_and(frame, mask_3ch)

    def process(self, frame):
        out = self.resize(frame)
        out = self.apply_mask(out)
        return out


# ─────────────────────────────────────────────────────────
# Visualize result
# ─────────────────────────────────────────────────────────
def draw_result(frame, boxes, confs, n_persons, inference_ms):
    """วาด bounding box + info บนภาพ"""
    img = frame.copy()
    h, w = img.shape[:2]

    for (x1, y1, x2, y2), conf in zip(boxes, confs):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 200, 80), 2)
        cv2.putText(img, f"person {conf:.2f}",
                    (int(x1), max(int(y1) - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 80),
                    1, cv2.LINE_AA)

    # info bar
    bar_h = 32
    bar_color = (0, 140, 60) if n_persons > 0 else (40, 40, 180)
    cv2.rectangle(img, (0, 0), (w, bar_h), bar_color, -1)
    info = f"persons={n_persons}  inference={inference_ms:.0f}ms"
    cv2.putText(img, info, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ─────────────────────────────────────────────────────────
# Run 1 image
# ─────────────────────────────────────────────────────────
def test_image(model, preprocessor, img_path, conf, out_dir, save_stages=True):
    img_path = Path(img_path)
    print(f"\n── {img_path.name} ──")

    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  [skip] cannot read image")
        return

    print(f"  Original size: {frame.shape[1]}x{frame.shape[0]}")

    # preprocess
    t0 = time.perf_counter()
    processed = preprocessor.process(frame)
    preprocess_ms = (time.perf_counter() - t0) * 1000
    print(f"  Processed size: {processed.shape[1]}x{processed.shape[0]}  "
          f"({preprocess_ms:.1f}ms)")

    # inference
    t0 = time.perf_counter()
    results = model(processed, conf=conf, classes=[0], verbose=False)
    inference_ms = (time.perf_counter() - t0) * 1000

    boxes, confs = [], []
    if results and results[0].boxes and len(results[0].boxes):
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        confs = results[0].boxes.conf.cpu().numpy().tolist()

    n = len(boxes)
    max_conf = max(confs) if confs else 0.0
    status = "OCCUPIED" if n > 0 else "empty"

    print(f"  Result: {status}  persons={n}  "
          f"max_conf={max_conf:.2f}  inference={inference_ms:.0f}ms")

    # save output
    annotated = draw_result(processed, boxes, confs, n, inference_ms)
    out_path = out_dir / f"{img_path.stem}_result.jpg"
    cv2.imwrite(str(out_path), annotated)

    if save_stages:
        # บันทึกภาพ intermediate ด้วย เพื่อดูว่า mask ทำงานไหม
        resized = preprocessor.resize(frame)
        cv2.imwrite(str(out_dir / f"{img_path.stem}_1_resized.jpg"), resized)
        if preprocessor.mask is not None:
            cv2.imwrite(str(out_dir / f"{img_path.stem}_2_masked.jpg"),
                        processed)

    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Test preprocessing + YOLO pipeline")
    parser.add_argument("--image", help="path to 1 test image")
    parser.add_argument("--folder", help="path to folder of test images")
    parser.add_argument("--mask", default=None, help="path to mask PNG (optional)")
    parser.add_argument("--model", default="yolo26s.pt", help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="confidence threshold")
    parser.add_argument("--resize", type=int, default=640,
                        help="resize width")
    parser.add_argument("--output", default="./test_output",
                        help="output folder")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="scan subfolders ด้วย")
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("ต้องระบุ --image หรือ --folder อย่างน้อย 1 อัน")

    # setup
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    print(f"conf={args.conf}  resize={args.resize}")

    preprocessor = Preprocessor(mask_path=args.mask, resize_width=args.resize)

    # gather images
    images = []
    if args.image:
        images.append(args.image)
    if args.folder:
        folder = Path(args.folder)
        if not folder.exists():
            print(f"[Error] Folder not found: {folder.resolve()}")
            return
        # รองรับทุกนามสกุลภาพ ไม่สนตัวพิมพ์ใหญ่/เล็ก
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        # recursive=True → หาในทุก subfolder ด้วย
        iterator = folder.rglob("*") if args.recursive else folder.iterdir()
        for f in sorted(iterator):
            if f.is_file() and f.suffix.lower() in exts:
                images.append(f)
        print(f"[Folder] Scanning {folder.resolve()}"
              f"{'  (recursive)' if args.recursive else ''}")
        print(f"[Folder] Found {len(images)} image(s)")

    if not images:
        print("No images found")
        return

    print(f"\nTesting {len(images)} image(s)...")
    print(f"Output folder: {out_dir.resolve()}")

    for img_path in images:
        test_image(model, preprocessor, img_path, args.conf, out_dir)

    print(f"\n✓ Done! Check results in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
