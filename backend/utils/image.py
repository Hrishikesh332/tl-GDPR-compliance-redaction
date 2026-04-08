import base64
import io

import cv2
import numpy as np
from PIL import Image


def load_image_from_bytes(data):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def crop_to_base64(img_bgr, box, fmt=".png", padding_ratio=0.0):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = img_bgr.shape[:2]
    if padding_ratio > 0:
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * padding_ratio)
        pad_y = int(bh * padding_ratio)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img_bgr[y1:y2, x1:x2]
    _, buf = cv2.imencode(fmt, crop)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def crop_face_to_base64(img_bgr, box, padding_ratio=0.4):
    """Crop a face with generous padding (40% default) and save as lossless PNG."""
    return crop_to_base64(img_bgr, box, fmt=".png", padding_ratio=padding_ratio)


def crop_with_bbox_to_base64(img_bgr, box, label=None, confidence=None,
                             fmt=".jpg", context_ratio=0.6):
    """Crop a region around the object with generous context and draw
    a bounding box + label. Gives the snapshot real visual context."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = img_bgr.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * context_ratio)
    pad_y = int(bh * context_ratio)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    if cx2 <= cx1 or cy2 <= cy1:
        return None

    crop = img_bgr[cy1:cy2, cx1:cx2].copy()

    rx1, ry1 = x1 - cx1, y1 - cy1
    rx2, ry2 = x2 - cx1, y2 - cy1

    color = (0, 220, 80)
    thickness = max(2, min(crop.shape[0], crop.shape[1]) // 150)
    cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), color, thickness)

    if label:
        text = label
        if confidence is not None:
            text = f"{label} {confidence:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(crop.shape[0], crop.shape[1]) / 800)
        txt_thick = max(1, int(font_scale * 2))
        (tw, th_text), baseline = cv2.getTextSize(text, font, font_scale, txt_thick)
        bg_y1 = max(0, ry1 - th_text - baseline - 6)
        bg_y2 = ry1
        cv2.rectangle(crop, (rx1, bg_y1), (rx1 + tw + 6, bg_y2), color, -1)
        cv2.putText(crop, text, (rx1 + 3, bg_y2 - baseline - 2),
                    font, font_scale, (0, 0, 0), txt_thick, cv2.LINE_AA)

    _, buf = cv2.imencode(fmt, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def apply_blur(frame, bbox, blur_strength=51):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    roi = frame[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]
    if roi_w <= 1 or roi_h <= 1:
        return frame

    strength = max(5, int(blur_strength))
    downscale_ratio = max(0.03, 0.24 - min(strength, 100) * 0.002)
    reduced_w = max(1, int(round(roi_w * downscale_ratio)))
    reduced_h = max(1, int(round(roi_h * downscale_ratio)))
    reduced = cv2.resize(roi, (reduced_w, reduced_h), interpolation=cv2.INTER_LINEAR)
    expanded = cv2.resize(reduced, (roi_w, roi_h), interpolation=cv2.INTER_CUBIC)

    kernel = max(9, min(max(roi_w, roi_h) | 1, (strength + 21) | 1))
    softened = cv2.GaussianBlur(expanded, (kernel, kernel), 0)
    frame[y1:y2, x1:x2] = softened
    return frame


def apply_pixelate(frame, bbox, pixel_size=12):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    roi = frame[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]
    down_w = max(1, roi_w // max(1, pixel_size))
    down_h = max(1, roi_h // max(1, pixel_size))
    reduced = cv2.resize(roi, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(reduced, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    return frame


def apply_black_fill(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    frame[y1:y2, x1:x2] = 0
    return frame


def apply_redaction(frame, bbox, mode="blur", blur_strength=51):
    mode_normalized = str(mode or "blur").strip().lower()
    if mode_normalized in {"solid", "black", "mask"}:
      return apply_black_fill(frame, bbox)
    if mode_normalized == "pixelate":
      return apply_pixelate(frame, bbox, pixel_size=max(6, blur_strength // 6))
    return apply_blur(frame, bbox, blur_strength)
