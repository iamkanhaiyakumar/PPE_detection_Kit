import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp"

from ultralytics import YOLO
import cv2
import math
import numpy as np
import torch

# ─── Shared Model & Config ────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def get_model_path():
    candidates = [
        os.path.join(BASE_DIR, "YOLO-Weights", "ppe.pt"),
        os.path.join(BASE_DIR, "ppe.pt"),
        os.path.join(BASE_DIR, "best.pt")
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return candidates[0]

CLASS_NAMES = [
    'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask',
    'NO-Safety Vest', 'Person', 'Safety Cone',
    'Safety Vest', 'machinery', 'vehicle'
]

COLOR_MAP = {
    'Hardhat': (0, 255, 0),
    'Mask': (0, 255, 0),
    'Safety Vest': (0, 255, 0),
    'NO-Hardhat': (0, 0, 255),
    'NO-Mask': (0, 0, 255),
    'NO-Safety Vest': (0, 0, 255),
    'machinery': (0, 149, 255),
    'vehicle': (0, 149, 255),
    'Person': (85, 45, 255),
    'Safety Cone': (85, 45, 255),
}

_model = None

def get_model():
    """Lazy-load or pre-load YOLO model singleton."""
    global _model
    if _model is None:
        path = get_model_path()
        print(f"🔄 Loading YOLO model from: {path}...")
        _model = YOLO(path)
        print("✅ YOLO model loaded successfully.")
    return _model


def _resize_if_needed(img, max_dim=720):
    """Resize image if max dimension exceeds max_dim to save RAM on Render (512MB limit)."""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        nw, nh = int(w * scale), int(h * scale)
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return img


def _draw_detections(img, results):
    """Draw prominent bounding boxes + styled label tags on img. Returns alert flag (True if NO-PPE detected)."""
    alert = False
    h, w = img.shape[:2]
    scale = max(w, h) / 750.0
    thickness = max(2, int(2.5 * scale))
    font_scale = max(0.55, 0.55 * scale)
    font_thickness = max(1, int(1.8 * scale))

    for r in results:
        for box in r.boxes:
            conf = round(float(box.conf[0]), 2)
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = CLASS_NAMES[class_id]
            label = f"{class_name} {conf}"
            color = COLOR_MAP.get(class_name, (85, 45, 255))

            if class_name in ('NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'):
                alert = True

            # Bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Filled label tag background
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            tag_y1 = max(0, y1 - th - baseline - 8)
            tag_y2 = y1
            tag_x2 = min(w, x1 + tw + 10)
            cv2.rectangle(img, (x1, tag_y1), (tag_x2, tag_y2), color, -1)

            # Label text
            text_y = tag_y2 - baseline - 3
            cv2.putText(img, label, (x1 + 5, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    return alert


# ─── Video Detection (generator for MJPEG streaming) ─────────────────────────

def video_detection(path_x):
    """
    Generator that yields annotated frames from a video file.
    path_x: str (file path)
    """
    print(f"🎥 Video Detection started for: {path_x}")

    if not isinstance(path_x, str) or not os.path.isfile(path_x):
        print(f"❌ Video file not found: {path_x}")
        return

    cap = cv2.VideoCapture(path_x)
    if not cap.isOpened():
        print(f"❌ Unable to open video source: {path_x}")
        return

    print("✅ Video opened successfully.")
    model = get_model()

    try:
        while True:
            success, img = cap.read()
            if not success:
                print("🚫 End of video stream.")
                break

            img = _resize_if_needed(img, max_dim=640)

            with torch.inference_mode():
                results = model(img, verbose=False, imgsz=640)

            _draw_detections(img, results)
            yield img
    except Exception as e:
        print(f"⚠️ Error during video stream: {e}")
    finally:
        cap.release()


# ─── Image Detection ──────────────────────────────────────────────────────────

def image_detection(image_path):
    """
    Run YOLO detection on a single image file.
    Returns: (annotated_img as numpy array, alert: bool)
    """
    print(f"🖼️ Image Detection started for: {image_path}")

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ Could not decode image: {image_path}")

    img = _resize_if_needed(img, max_dim=720)
    model = get_model()

    with torch.inference_mode():
        results = model(img, verbose=False, imgsz=640)

    alert = _draw_detections(img, results)
    print(f"✅ Image detection complete. Alert: {alert}")
    return img, alert


# ─── Webcam Frame Detection (single frame, browser-based) ────────────────────

def process_single_frame(frame_bytes):
    """
    Run YOLO detection on a single JPEG frame (bytes from browser webcam).
    Returns: (annotated JPEG bytes, alert: bool)
    """
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("❌ Could not decode frame bytes.")

    img = _resize_if_needed(img, max_dim=640)
    model = get_model()

    with torch.inference_mode():
        results = model(img, verbose=False, imgsz=640)

    alert = _draw_detections(img, results)

    # Encode back to JPEG
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buffer.tobytes(), alert
