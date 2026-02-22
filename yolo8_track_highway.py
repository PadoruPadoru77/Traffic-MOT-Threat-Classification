import argparse
import time
from pathlib import Path
import torch

import cv2
import numpy as np
from ultralytics import YOLO

import supervision as sv


# COCO ids: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_IDS = {2, 3, 5, 7}


def parse_args():
    p = argparse.ArgumentParser("YOLOv8 tiled detection + ByteTrack")
    p.add_argument("--video", type=str, default="highway.mp4")
    p.add_argument("--model", type=str, default="yolov8m.pt")
    p.add_argument("--out", type=str, default="tracked_tiled.mp4")
    p.add_argument("--conf", type=float, default=0.07)
    p.add_argument("--iou", type=float, default=0.6)

    # Tiling params
    p.add_argument("--tile", type=int, default=512, help="Tile size (square). 640 is a good start.")
    p.add_argument("--overlap", type=float, default=0.30, help="Tile overlap ratio (0.15–0.30 typical).")

    # Optional: only keep vehicles
    p.add_argument("--vehicles_only", action="store_true", help="Filter to car/truck/bus/motorcycle")

    # Device
    p.add_argument("--device", type=str, default="cuda", help="cuda, cuda:0, cpu")
    return p.parse_args()


def iter_tiles(frame_w, frame_h, tile, overlap):
    """Yield tile rectangles (x1,y1,x2,y2) covering the full frame with overlap."""
    step = int(tile * (1.0 - overlap))
    step = max(step, 1)

    xs = list(range(0, max(frame_w - tile, 0) + 1, step))
    ys = list(range(0, max(frame_h - tile, 0) + 1, step))

    # Ensure last tile reaches the far edge
    if xs[-1] != frame_w - tile:
        xs.append(max(frame_w - tile, 0))
    if ys[-1] != frame_h - tile:
        ys.append(max(frame_h - tile, 0))

    for y in ys:
        for x in xs:
            x1, y1 = x, y
            x2, y2 = min(x + tile, frame_w), min(y + tile, frame_h)
            yield x1, y1, x2, y2


def nms_xyxy(boxes, scores, iou_thresh):
    """
    Simple class-agnostic NMS for xyxy boxes.
    boxes: (N,4) float
    scores: (N,) float
    returns indices kept
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=int)


def tiled_detect(model, frame, tile, overlap, conf, iou, vehicles_only):
    """
    Run YOLO on overlapping tiles, merge detections into full-frame coords, then NMS.
    Returns: xyxy (N,4), scores (N,), class_ids (N,)
    """
    H, W = frame.shape[:2]
    all_boxes = []
    all_scores = []
    all_clss = []

    # Run detection per tile
    for x1, y1, x2, y2 in iter_tiles(W, H, tile, overlap):
        tile_img = frame[y1:y2, x1:x2]

        # YOLO inference on tile
        r = model.predict(tile_img, conf=conf, iou=iou, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        if vehicles_only:
            mask = np.array([c in VEHICLE_IDS for c in clss], dtype=bool)
            boxes, scores, clss = boxes[mask], scores[mask], clss[mask]
            if len(boxes) == 0:
                continue

        # Offset tile coords back to full-frame coords
        boxes[:, [0, 2]] += x1
        boxes[:, [1, 3]] += y1

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_clss.append(clss)

    if not all_boxes:
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=int)

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    clss = np.concatenate(all_clss, axis=0)

    # Do NMS per class (better than class-agnostic)
    keep_indices = []
    for c in np.unique(clss):
        idx = np.where(clss == c)[0]
        kept = nms_xyxy(boxes[idx], scores[idx], iou_thresh=0.5)  # merge duplicates from overlapping tiles
        keep_indices.append(idx[kept])

    keep_indices = np.concatenate(keep_indices, axis=0) if keep_indices else np.array([], dtype=int)
    keep_indices = keep_indices[scores[keep_indices].argsort()[::-1]]

    return boxes[keep_indices], scores[keep_indices], clss[keep_indices]


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    model = YOLO(args.model)
    # Force device
    try:
        model.to(args.device)
    except Exception:
        # If user passed "cuda" and it fails, fall back to cpu
        model.to("cpu")
    device_used = next(model.model.parameters()).device
    print(f"Model is running on device: {device_used}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # ByteTrack tracker (for stable IDs)
    tracker = sv.ByteTrack()

    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        boxes, scores, clss = tiled_detect(
            model=model,
            frame=frame,
            tile=args.tile,
            overlap=args.overlap,
            conf=args.conf,
            iou=args.iou,
            vehicles_only=args.vehicles_only
        )

        # Build supervision detections
        detections = sv.Detections(
            xyxy=boxes.astype(np.float32),
            confidence=scores.astype(np.float32),
            class_id=clss.astype(int)
        )

        # Update tracker -> adds track IDs
        tracked = tracker.update_with_detections(detections)

        # Draw
        for xyxy, conf, cls_id, track_id in zip(
            tracked.xyxy, tracked.confidence, tracked.class_id, tracked.tracker_id
        ):
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {int(track_id)} cls {int(cls_id)} {float(conf):.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    dt = time.time() - t0
    print(f"Done. Wrote {out_path}")
    print(f"Frames: {frame_idx}  Time: {dt:.2f}s  FPS: {frame_idx / dt:.2f}")


if __name__ == "__main__":
    main()