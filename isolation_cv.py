"""CV isolation helpers.

Uses multiple classical methods (Otsu threshold, HSV saturation, Canny edges) to propose
candidate masks/bboxes, scores them, de-duplicates, and returns the best ones.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional

# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
StepImage = Tuple[str, np.ndarray]  # (step_name, RGB image for display)
BBox = Tuple[int, int, int, int]    # (x, y, width, height)


# -----------------------------------------------------------------------------
# Image conversion and morphology helpers
# -----------------------------------------------------------------------------

def _to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) image to RGB for display."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert grayscale mask to 3-channel RGB for visualization."""
    if mask.ndim == 2:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return mask


def _morph_clean(mask: np.ndarray, k_open: int = 3, k_close: int = 7) -> np.ndarray:
    """Binarize mask and apply open then close to remove noise and fill holes."""
    mask = (mask > 0).astype(np.uint8) * 255
    if k_open > 0:
        k = np.ones((k_open, k_open), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if k_close > 0:
        k = np.ones((k_close, k_close), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


# -----------------------------------------------------------------------------
# Mask generation: Otsu, HSV, Canny
# -----------------------------------------------------------------------------

def _masks_from_otsu_both_polarities(image_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Return both polarities so full object can win over bright/dark patch. No 85% heuristic."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m1 = _morph_clean(m, k_open=3, k_close=9)
    m2 = _morph_clean(cv2.bitwise_not(m), k_open=3, k_close=9)
    return [("otsu", m1), ("otsu_inv", m2)]


def _mask_from_hsv_saturation(image_bgr: np.ndarray, sat_thresh: int = 40) -> np.ndarray:
    """Segment by saturation: low-saturation (e.g. white/gray) vs colored object."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    m = (s >= sat_thresh).astype(np.uint8) * 255
    return _morph_clean(m, k_open=3, k_close=11)


def _auto_canny_thresholds(gray: np.ndarray) -> Tuple[int, int]:
    """Compute low/high Canny thresholds from median intensity."""
    med = float(np.median(gray))
    low = int(max(10, 0.66 * med))
    high = int(min(220, 1.33 * med))
    if high <= low:
        high = min(255, low + 40)
    return low, high


def _mask_from_canny(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Canny edges + close + fill contours; returns (edge map, filled binary mask)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    low, high = _auto_canny_thresholds(gray_blur)
    edges = cv2.Canny(gray_blur, low, high)
    # Close edges to form blobs; then fill contours into a binary mask
    k = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(closed)
    if contours:
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    mask = _morph_clean(mask, k_open=0, k_close=9)
    return edges, mask


def _bboxes_from_mask(mask: np.ndarray, min_area_px: int) -> List[Tuple[BBox, np.ndarray, float]]:
    """Extract bounding box, contour, and area for each contour above min_area_px."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Tuple[BBox, np.ndarray, float]] = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area_px:
            continue
        x, y, w, h = cv2.boundingRect(c)
        out.append(((int(x), int(y), int(w), int(h)), c, area))
    return out


def _is_canny_informative(edges: np.ndarray, min_edge_frac: float = 0.008) -> bool:
    """True if edge map has enough structure to use edge_density in scoring."""
    if edges is None or edges.size == 0:
        return False
    return float((edges > 0).sum()) / float(edges.size) >= min_edge_frac


def _bbox_area(bbox: BBox) -> float:
    """Return area of bounding box (width * height)."""
    return float(bbox[2] * bbox[3])


def _is_inside(inner: BBox, outer: BBox) -> bool:
    """True if inner bbox is completely inside outer (and strictly smaller)."""
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    if (ix, iy, iw, ih) == (ox, oy, ow, oh):
        return False
    return (
        ix >= ox
        and iy >= oy
        and ix + iw <= ox + ow
        and iy + ih <= oy + oh
    )


def _drop_fragments(picked: List[BBox], overlap_thresh: float = 0.5, area_ratio: float = 2.0) -> List[BBox]:
    """Remove boxes that are largely inside a much larger box (fragment) or fully contained in another (e.g. hole)."""
    if len(picked) <= 1:
        return picked
    keep = []
    for b in picked:
        area_b = _bbox_area(b)
        # Fragment: larger box overlaps this one a lot
        is_fragment = any(
            _bbox_area(p) >= area_ratio * area_b and _iou(b, p) >= overlap_thresh
            for p in picked
            if p is not b
        )
        # Contained: this box is completely inside another (e.g. hole inside circle LEGO) -> drop smaller
        is_contained = any(_is_inside(b, p) for p in picked if p is not b)
        if not is_fragment and not is_contained:
            keep.append(b)
    return keep


def _iou(a: BBox, b: BBox) -> float:
    """Intersection over Union of two axis-aligned boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter) / float(union + 1e-9)


def _score_candidate(
    bbox: BBox,
    contour: np.ndarray,
    contour_area: float,
    image_shape: Tuple[int, int],
    edges: np.ndarray,
    gray: np.ndarray,
    method_name: str,
    canny_informative: bool,
) -> float:
    h_img, w_img = image_shape
    x, y, w, h = bbox
    img_area = float(h_img * w_img)

    # Size preferences (reject entire-picture blobs > 0.85; allow single big object but not full frame)
    area_frac = contour_area / (img_area + 1e-9)
    if area_frac < 0.001 or area_frac > 0.85:
        return -1e9

    # Reject when bbox covers almost the whole image (full-frame crop even if contour area is slightly under cap)
    bbox_area = float(w * h) + 1e-9
    bbox_frac = bbox_area / (img_area + 1e-9)
    if bbox_frac > 0.85:
        return -1e9

    # Solidity (filledness vs spiky/noisy contours)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull)) + 1e-9
    solidity = float(contour_area) / hull_area

    # Aspect ratio preference (avoid extreme thin strips)
    ar = float(w) / float(h + 1e-9)
    ar_penalty = 0.0
    if ar < 0.15 or ar > 6.5:
        ar_penalty = 0.5

    # Border-touch penalty (softer: object on white table often touches edge)
    border_touch = 0.3 if (x <= 0 or y <= 0 or (x + w) >= (w_img - 1) or (y + h) >= (h_img - 1)) else 0.0

    # Edge density inside bbox (only use when Canny is informative; else it skews toward blank blobs)
    roi_edges = edges[y : y + h, x : x + w]
    edge_density = float((roi_edges > 0).mean()) if roi_edges.size else 0.0
    edge_weight = 1.2 if canny_informative else 0.0
    score = 2.0 * solidity + edge_weight * edge_density - border_touch - ar_penalty

    # Prefer Otsu when Canny is dead (Otsu was right, HSV gave blank and won otherwise)
    if not canny_informative and method_name in ("otsu", "otsu_inv"):
        score += 0.25

    # Penalize blank/largely blank background (including crops with a bit of shadow)
    if gray is not None and gray.size > 0:
        roi_gray = gray[y : y + h, x : x + w]
        if roi_gray.size >= 4:
            mean_val = float(roi_gray.mean())
            var_val = float(roi_gray.var()) if roi_gray.size else 0.0
            white_frac = float((roi_gray > 235).sum()) / float(roi_gray.size)
            if mean_val > 240 and var_val < 120 and edge_density < 0.02:
                score -= 1.5  # very blank
            elif mean_val > 230 and var_val < 300 and (edge_density < 0.05 or white_frac > 0.75):
                score -= 1.2  # largely blank (e.g. white + shadow)

    # Extent (tightness): prefer tight box around object; fragments often have lower contour fill
    extent = contour_area / bbox_area
    score += 0.5 * extent

    # Area bonus: full object often beats fragment when scores are close (Case B)
    score += 0.4 * min(area_frac, 0.5)

    return float(score)


# -----------------------------------------------------------------------------
# Internal: build candidates and pick bboxes (shared by get_cv_bboxes and get_cv_bboxes_and_steps)
# -----------------------------------------------------------------------------

def _get_cv_bboxes_internal(
    image: np.ndarray,
    max_boxes: int = 10,
    iou_dedupe: float = 0.35,
    min_score_threshold: Optional[float] = None,
    precomputed: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, np.ndarray]]]] = None,
) -> Tuple[List[BBox], List[Tuple[float, str, BBox]]]:
    """Build masks (or use precomputed), score candidates, pick and de-duplicate. Returns (picked, candidates)."""
    h_img, w_img = image.shape[:2]
    min_area_px = int(0.001 * (h_img * w_img))

    if precomputed is not None:
        gray, edges, canny_mask, masks = precomputed
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges, canny_mask = _mask_from_canny(image)
        masks = list(_masks_from_otsu_both_polarities(image))
        masks.append(("hsv_sat", _mask_from_hsv_saturation(image)))
        masks.append(("canny_fill", canny_mask))

    canny_informative = _is_canny_informative(edges)

    candidates: List[Tuple[float, str, BBox]] = []
    for name, m in masks:
        for bbox, contour, area in _bboxes_from_mask(m, min_area_px=min_area_px):
            score = _score_candidate(
                bbox, contour, area, (h_img, w_img), edges, gray, name, canny_informative
            )
            if score <= -1e8:
                continue
            candidates.append((score, name, bbox))

    if not candidates:
        return [], candidates

    candidates.sort(key=lambda t: t[0], reverse=True)
    picked: List[BBox] = []
    for score, name, bbox in candidates:
        if min_score_threshold is not None and score < min_score_threshold:
            continue
        if any(_iou(bbox, pb) >= iou_dedupe for pb in picked):
            continue
        picked.append(bbox)
        if len(picked) >= max_boxes:
            break
    picked = _drop_fragments(picked)
    return picked, candidates


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def get_cv_bboxes(
    image: np.ndarray,
    max_boxes: int = 10,
    padding: int = 0,
    iou_dedupe: float = 0.35,
    min_score_threshold: Optional[float] = None,
) -> List[BBox]:
    """
    Robust classical isolation.
    - Generate candidate masks/bboxes from: Otsu both polarities, HSV saturation, Canny.
    - Score candidates (solidity, edge density when Canny informative, blank penalty, area bonus).
    - Only accept candidates with score >= min_score_threshold (if set).
    - De-duplicate overlapping bboxes and return up to max_boxes.
    """
    picked, _ = _get_cv_bboxes_internal(
        image, max_boxes=max_boxes, iou_dedupe=iou_dedupe,
        min_score_threshold=min_score_threshold,
    )
    if not picked:
        return []
    h_img, w_img = image.shape[:2]
    if padding > 0:
        out: List[BBox] = []
        for x, y, w, h in picked:
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w_img, x + w + padding)
            y2 = min(h_img, y + h + padding)
            out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        return out
    return picked


def crop_from_bbox(image: np.ndarray, bbox: BBox, padding: int = 0) -> np.ndarray:
    """Return image crop for bbox (x, y, w, h), with optional padding. Clips to image bounds."""
    x, y, w, h = bbox
    h_img, w_img = image.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)
    return image[y1:y2, x1:x2].copy()


def get_cv_bboxes_and_steps(
    image: np.ndarray,
    padding: int = 10,
    max_boxes: int = 10,
    min_score_threshold: Optional[float] = None,
) -> Tuple[List[Tuple[int, int, int, int]], List[StepImage]]:
    """
    Same as get_cv_bboxes but also return visualization images for each step.
    Builds masks once and reuses them for bbox picking. Returns (bboxes, steps).
    """
    steps: List[StepImage] = []
    rgb = _to_rgb(image)
    steps.append(("1. Original", rgb.copy()))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    steps.append(("2. Grayscale", _mask_to_rgb(gray)))

    edges, canny_mask = _mask_from_canny(image)
    steps.append(("3. Edges (Canny, auto)", _mask_to_rgb(edges)))
    steps.append(("4. Mask (Canny filled)", _mask_to_rgb(canny_mask)))

    otsu_masks = _masks_from_otsu_both_polarities(image)
    steps.append(("5. Mask (Otsu)", _mask_to_rgb(otsu_masks[0][1])))
    steps.append(("5b. Mask (Otsu inv)", _mask_to_rgb(otsu_masks[1][1])))

    hsv_mask = _mask_from_hsv_saturation(image)
    steps.append(("6. Mask (HSV saturation)", _mask_to_rgb(hsv_mask)))

    masks = list(otsu_masks) + [("hsv_sat", hsv_mask), ("canny_fill", canny_mask)]
    bboxes, _ = _get_cv_bboxes_internal(
        image,
        max_boxes=max_boxes,
        iou_dedupe=0.35,
        min_score_threshold=min_score_threshold,
        precomputed=(gray, edges, canny_mask, masks),
    )
    bbox_img = rgb.copy()
    for (x, y, w, h) in bboxes:
        cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    steps.append(("7. Selected bboxes", bbox_img))

    for i, bbox in enumerate(bboxes):
        crop_bgr = crop_from_bbox(image, bbox, padding=padding)
        steps.append((f"8. Crop {i + 1}", _to_rgb(crop_bgr)))

    return bboxes, steps
