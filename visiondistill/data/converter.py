from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Segmentation: mask → YOLO polygon labels
# ---------------------------------------------------------------------------


def mask_to_yolo_polygons(
    mask: np.ndarray,
    class_id: int = 0,
    epsilon_ratio: float = 0.002,
    min_area: int = 100,
) -> list[str]:
    """Convert a binary mask to YOLO segmentation label lines.

    Each returned string is one line: ``class_id x1 y1 x2 y2 ... xn yn``
    with coordinates normalized to [0, 1].
    """
    h, w = mask.shape[:2]
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines: list[str] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue

        coords = approx.squeeze()
        if coords.ndim != 2 or coords.shape[1] != 2:
            continue

        norm_coords = coords.astype(np.float64)
        norm_coords[:, 0] /= w
        norm_coords[:, 1] /= h
        np.clip(norm_coords, 0.0, 1.0, out=norm_coords)

        flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm_coords)
        lines.append(f"{class_id} {flat}")

    return lines


def masks_to_label_file(
    masks: np.ndarray,
    class_ids: list[int] | None = None,
    epsilon_ratio: float = 0.002,
    min_area: int = 100,
) -> str:
    """Convert N binary masks to a full YOLO label file content string."""
    if class_ids is None:
        class_ids = [0] * len(masks)

    all_lines: list[str] = []
    for mask, cid in zip(masks, class_ids):
        all_lines.extend(
            mask_to_yolo_polygons(mask, class_id=cid, epsilon_ratio=epsilon_ratio, min_area=min_area)
        )
    return "\n".join(all_lines)


# ---------------------------------------------------------------------------
# Detection: xyxy boxes → YOLO detect labels  (class cx cy w h, normalised)
# ---------------------------------------------------------------------------


def boxes_xyxy_to_yolo_detect_lines(
    boxes: np.ndarray,
    class_ids: list[int],
    image_w: int,
    image_h: int,
) -> list[str]:
    """Convert (N, 4) xyxy pixel-coordinate boxes to YOLO detect label lines.

    Returns one string per box: ``class_id cx cy w h`` with all values
    normalised to [0, 1].
    """
    lines: list[str] = []
    for (x1, y1, x2, y2), cid in zip(boxes, class_ids):
        cx = ((x1 + x2) / 2.0) / image_w
        cy = ((y1 + y2) / 2.0) / image_h
        bw = (x2 - x1) / image_w
        bh = (y2 - y1) / image_h
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def boxes_to_yolo_label_file(
    boxes: np.ndarray,
    class_ids: list[int] | None = None,
    image_w: int = 1,
    image_h: int = 1,
) -> str:
    """Convert (N, 4) xyxy boxes to a YOLO detect label file string."""
    if class_ids is None:
        class_ids = [0] * len(boxes)
    return "\n".join(
        boxes_xyxy_to_yolo_detect_lines(boxes, class_ids, image_w, image_h)
    )
