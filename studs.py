"""ROI, step-image parsing, stud detection, and rule-based LEGO classification."""
import cv2
import numpy as np
from typing import List, Tuple, Optional

# Hough circle parameters for stud detection
HC_DP = 1.2
HC_MIN_DIST = 22
HC_PARAM1 = 80
HC_PARAM2 = 18
HC_MIN_RADIUS = 8
HC_MAX_RADIUS = 22


def get_center_roi(
    frame: np.ndarray,
    roi_w_ratio: float = 0.55,
    roi_h_ratio: float = 0.55,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Extract center ROI from frame. Returns (roi_image, (x0, y0, x1, y1))."""
    H, W = frame.shape[:2]
    roi_w = int(W * roi_w_ratio)
    roi_h = int(H * roi_h_ratio)
    x0 = (W - roi_w) // 2
    y0 = (H - roi_h) // 2
    x1 = x0 + roi_w
    y1 = y0 + roi_h
    roi = frame[y0:y1, x0:x1].copy()
    return roi, (x0, y0, x1, y1)


def parse_step_images(
    step_images: List[Tuple[str, np.ndarray]],
) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
    """Parse step_images from get_cv_bboxes_and_steps. Returns (selected_box_rgb, crop_rgbs)."""
    selected_box_rgb = None
    crop_rgbs = []
    if not step_images:
        return selected_box_rgb, crop_rgbs
    for item in step_images:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        name, rgb = item
        name_low = str(name).lower()
        if "selected bboxes" in name_low or "selected bbox" in name_low:
            selected_box_rgb = rgb
        if "crop" in name_low:
            crop_rgbs.append(rgb)
    return selected_box_rgb, crop_rgbs


def contour_circularity_from_crop(crop_bgr: np.ndarray) -> float:
    """Compute contour circularity (0–1) for a BGR crop."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    h, w = crop_bgr.shape[:2]
    if h < 11 or w < 11:
        return 0.0
    try:
        gray = cv2.cvtColor(np.ascontiguousarray(crop_bgr), cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(np.ascontiguousarray(gray), (5, 5), 0)
    except cv2.error:
        return 0.0
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(th == 255) > 0.5:
        th = 255 - th
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri + 1e-6))


def _deduplicate_circles(circles: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    if not circles:
        return []
    dedup = []
    circles = sorted(circles, key=lambda t: t[2], reverse=True)
    for c in circles:
        x, y, r = c
        keep = True
        for x2, y2, r2 in dedup:
            center_dist = np.hypot(x - x2, y - y2)
            radius_diff = abs(r - r2)
            if center_dist < 0.55 * max(r, r2) and radius_diff < 0.45 * max(r, r2):
                keep = False
                break
        if keep:
            dedup.append(c)
    return dedup


def _detect_center_hole_strict(blur: np.ndarray, vis: np.ndarray) -> bool:
    H, W = blur.shape[:2]
    cx0, cy0 = W // 2, H // 2
    roi_w = int(W * 0.28)
    roi_h = int(H * 0.28)
    x0 = max(0, cx0 - roi_w // 2)
    y0 = max(0, cy0 - roi_h // 2)
    x1 = min(W, cx0 + roi_w // 2)
    y1 = min(H, cy0 + roi_h // 2)
    center_roi = blur[y0:y1, x0:x1]
    if center_roi.size == 0:
        return False
    _, th_dark = cv2.threshold(center_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = np.ones((3, 3), np.uint8)
    th_dark = cv2.morphologyEx(th_dark, cv2.MORPH_OPEN, k, iterations=1)
    th_dark = cv2.morphologyEx(th_dark, cv2.MORPH_CLOSE, k, iterations=1)
    cnts, _ = cv2.findContours(th_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = center_roi.shape[0] * center_roi.shape[1]
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 25 or a > 0.18 * roi_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / (h + 1e-6)
        cx = x + w / 2
        cy = y + h / 2
        if not (0.30 * center_roi.shape[1] <= cx <= 0.70 * center_roi.shape[1]):
            continue
        if not (0.30 * center_roi.shape[0] <= cy <= 0.70 * center_roi.shape[0]):
            continue
        if not (0.60 <= aspect <= 1.60):
            continue
        cv2.rectangle(vis, (x + x0, y + y0), (x + x0 + w, y + y0 + h), (255, 0, 0), 2)
        return True
    return False


def detect_studs_and_center_hole(
    crop_bgr: np.ndarray,
) -> Tuple[int, List[Tuple[int, int, int]], Optional[np.ndarray], bool]:
    """Detect studs (Hough circles) and center hole on a BGR crop. Returns (stud_count, circles, vis_image, center_hole_detected)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0, [], None, False
    h, w = crop_bgr.shape[:2]
    if h < 11 or w < 11:
        return 0, [], np.ascontiguousarray(crop_bgr).copy(), False
    crop_bgr = np.ascontiguousarray(crop_bgr)
    if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
        return 0, [], crop_bgr.copy(), False
    vis = crop_bgr.copy()
    try:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = np.ascontiguousarray(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
    except cv2.error:
        return 0, [], vis, False
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=HC_DP,
        minDist=HC_MIN_DIST,
        param1=HC_PARAM1,
        param2=HC_PARAM2,
        minRadius=HC_MIN_RADIUS,
        maxRadius=HC_MAX_RADIUS,
    )
    kept = []
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        H, W = blur.shape[:2]
        for x, y, r in circles:
            if x - r < 2 or y - r < 2 or x + r >= W - 2 or y + r >= H - 2:
                continue
            mask_in = np.zeros_like(blur, dtype=np.uint8)
            cv2.circle(mask_in, (x, y), r, 255, -1)
            mask_out = np.zeros_like(blur, dtype=np.uint8)
            cv2.circle(mask_out, (x, y), int(r * 1.45), 255, -1)
            ring = cv2.subtract(mask_out, mask_in)
            inside_vals = blur[mask_in == 255]
            ring_vals = blur[ring == 255]
            if len(inside_vals) == 0 or len(ring_vals) == 0:
                continue
            inside_mean = float(np.mean(inside_vals))
            ring_mean = float(np.mean(ring_vals))
            if abs(inside_mean - ring_mean) < 5.5:
                continue
            kept.append((x, y, r))
    dedup = _deduplicate_circles(kept)
    center_hole_detected = _detect_center_hole_strict(blur, vis)
    for x, y, r in dedup:
        cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
        cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)
    return len(dedup), dedup, vis, center_hole_detected


def classify_from_studs_only(stud_count: int, center_hole_detected: bool) -> str:
    """Rule-based label from stud count (and optionally center hole). Returns 2x1, rec, cir, squ, or unknown."""
    if stud_count == 2:
        return "2x1"
    if stud_count >= 6:
        return "rec"
    if stud_count == 5:
        return "cir"
    if 3 <= stud_count <= 4:
        return "squ"
    return "unknown"
