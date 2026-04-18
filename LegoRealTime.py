"""LEGO real-time detection: camera → ROI → isolation → stud detection → rule classify → dashboard."""
import sys
import time
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess
from src.isolation_cv import get_cv_bboxes_and_steps
from src.studs import (
    get_center_roi,
    parse_step_images,
    contour_circularity_from_crop,
    detect_studs_and_center_hole,
    classify_from_studs_only,
)


# =====================
# Config
# =====================
CAM_INDEX = 0
MIN_SCORE_THRESHOLD = 2.0
PREPROCESS_MAX_SIDE = 1024
FRAME_W = 1280
FRAME_H = 720
PANEL_W = 640
PANEL_H = 480
ROI_W_RATIO = 0.55
ROI_H_RATIO = 0.55
DRAW_ROI_BOX = True
MAX_OBJECTS = 6
FPS_ALPHA = 0.9


# =====================
# UI helpers
# =====================
def resize_to_fit(img, target_w, target_h):
    if img is None or img.size == 0:
        return np.full((target_h, target_w, 3), 30, dtype=np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_h, target_w, 3), 30, dtype=np.uint8)
    scale = min(target_w / w, target_h / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), 30, dtype=np.uint8)
    x0 = (target_w - nw) // 2
    y0 = (target_h - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def add_title(img, title):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 38), (45, 45, 45), -1)
    cv2.putText(out, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_crop_grid(crop_images, target_w, target_h):
    canvas = np.full((target_h, target_w, 3), 30, dtype=np.uint8)
    if not crop_images:
        return canvas
    n = min(len(crop_images), 4)
    if n == 1:
        return resize_to_fit(crop_images[0], target_w, target_h)
    cols, rows = 2, int(np.ceil(n / 2))
    cell_w, cell_h = target_w // cols, target_h // rows
    for idx in range(n):
        r, c = idx // cols, idx % cols
        cell = resize_to_fit(crop_images[idx], cell_w, cell_h)
        y0, x0 = r * cell_h, c * cell_w
        canvas[y0 : y0 + cell_h, x0 : x0 + cell_w] = cell
    return canvas


def make_text_panel(objects_info, fps, w, h):
    panel = np.full((h, w, 3), 20, dtype=np.uint8)

    def put(txt, x, y, color=(220, 220, 220), scale=0.68, thick=2):
        cv2.putText(panel, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    y = 40
    put(f"FPS: {fps:.1f}", 20, y, (255, 255, 0), 0.9, 2)
    y += 40
    put(f"Detected objects: {len(objects_info)}", 20, y, (255, 255, 255), 0.85, 2)
    y += 38
    if not objects_info:
        put("No objects detected", 20, y, (0, 180, 255), 0.8, 2)
        return panel
    for i, info in enumerate(objects_info[:5]):
        put(f"Obj {i+1}: {info['final_label']}", 20, y, (0, 255, 0), 0.85, 2)
        y += 28
        put(f"  studs={info['stud_count']}", 35, y, (220, 220, 220), 0.60, 1)
        y += 22
        put(f"  hole={info['center_hole_detected']}", 35, y, (220, 220, 220), 0.60, 1)
        y += 22
        put(f"  crop_circ={info['crop_circularity']:.3f}", 35, y, (220, 220, 220), 0.60, 1)
        y += 26
        if y > h - 25:
            break
    return panel


# =====================
# Main
# =====================
def main():
    print("Starting LEGO real-time detection...")
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open camera (index %s). Check that a camera is connected and not in use." % CAM_INDEX)
        input("Press Enter to exit...")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    prev_t = time.time()
    fps = 0.0
    print("Press q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 1e-6:
                inst_fps = 1.0 / dt
                fps = inst_fps if fps == 0 else FPS_ALPHA * fps + (1 - FPS_ALPHA) * inst_fps

            camera_view = frame.copy()
            selected_box_view = frame.copy()
            crop_images = []
            objects_info = []

            roi, (rx0, ry0, rx1, ry1) = get_center_roi(frame, roi_w_ratio=ROI_W_RATIO, roi_h_ratio=ROI_H_RATIO)
            if DRAW_ROI_BOX:
                cv2.rectangle(camera_view, (rx0, ry0), (rx1, ry1), (255, 255, 0), 2)
                cv2.rectangle(selected_box_view, (rx0, ry0), (rx1, ry1), (255, 255, 0), 2)

            try:
                cv_img = preprocess(roi, max_side=PREPROCESS_MAX_SIDE)
                _, step_images = get_cv_bboxes_and_steps(
                    cv_img,
                    padding=10,
                    max_boxes=MAX_OBJECTS,
                    min_score_threshold=MIN_SCORE_THRESHOLD,
                )
                if step_images:
                    step_images[0] = ("1. Original (raw)", cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            except Exception:
                step_images = []

            selected_box_rgb, crop_rgbs = parse_step_images(step_images)
            if selected_box_rgb is not None:
                selected_box_bgr = cv2.cvtColor(selected_box_rgb, cv2.COLOR_RGB2BGR)
                overlay = selected_box_view.copy()
                roi_vis = resize_to_fit(selected_box_bgr, rx1 - rx0, ry1 - ry0)
                overlay[ry0:ry1, rx0:rx1] = roi_vis
                selected_box_view = overlay

            for crop_rgb in crop_rgbs[:MAX_OBJECTS]:
                crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                stud_count, _, stud_vis, center_hole_detected = detect_studs_and_center_hole(crop_bgr)
                crop_circ = contour_circularity_from_crop(crop_bgr)
                final_label = classify_from_studs_only(stud_count, center_hole_detected)
                crop_images.append(stud_vis if stud_vis is not None else crop_bgr)
                objects_info.append({
                    "final_label": final_label,
                    "stud_count": stud_count,
                    "center_hole_detected": center_hole_detected,
                    "crop_circularity": crop_circ,
                })

            panel_tl = add_title(resize_to_fit(camera_view, PANEL_W, PANEL_H), "Camera")
            panel_tr = add_title(resize_to_fit(selected_box_view, PANEL_W, PANEL_H), "Selected Box")
            panel_bl = add_title(make_crop_grid(crop_images, PANEL_W, PANEL_H), "Crop + Studs")
            panel_br = add_title(make_text_panel(objects_info, fps, PANEL_W, PANEL_H), "Prediction Data")
            dashboard = np.vstack([
                np.hstack([panel_tl, panel_tr]),
                np.hstack([panel_bl, panel_br]),
            ])
            cv2.imshow("Realtime LEGO Detection Dashboard", dashboard)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
