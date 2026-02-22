import argparse
import time
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


# COCO ids: car=2, motorcycle=3, bus=5, truck=7
DEFAULT_VEHICLE_IDS = {2, 3, 5, 7}


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 vehicle detection + tracking on a video.")
    p.add_argument("--video", type=str, required=True, help="Path to input video (mp4, mov, etc.)")
    p.add_argument("--model", type=str, default="yolov8s.pt", help="YOLOv8 model path (e.g., yolov8n.pt, yolov8s.pt)")
    p.add_argument("--out", type=str, default="tracked_output.mp4", help="Path to output annotated video")
    p.add_argument("--csv", type=str, default="tracks.csv", help="Path to output CSV of tracklets (set '' to disable)")
    p.add_argument("--imgsz", type=int, default=960, help="Inference image size (bigger helps small cars)")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--device", type=str, default=None, help="cuda / cpu / 0 / 1 ... (default: auto)")
    p.add_argument("--classes", type=str, default="car,truck,bus,motorcycle",
                   help="Comma-separated classes to keep (car,truck,bus,motorcycle) or 'all'")
    p.add_argument("--tracker", type=str, default="bytetrack.yaml",
                   help="Tracker config: bytetrack.yaml (default) or botsort.yaml")
    return p.parse_args()


def class_name_to_coco_id(name: str):
    # COCO names in Ultralytics for relevant vehicle classes
    mapping = {
        "car": 2,
        "motorcycle": 3,
        "bus": 5,
        "truck": 7,
    }
    return mapping.get(name.lower().strip())


def get_vehicle_class_ids(classes_arg: str):
    if classes_arg.strip().lower() == "all":
        return None  # keep all
    ids = set()
    for token in classes_arg.split(","):
        cid = class_name_to_coco_id(token)
        if cid is not None:
            ids.add(cid)
    # fall back if user typo'd
    return ids if ids else DEFAULT_VEHICLE_IDS


def main():
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    keep_ids = get_vehicle_class_ids(args.classes)

    # Load model
    model = YOLO(args.model)

    # Open video for metadata
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Video writer
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # Storage for CSV
    rows = []

    # Run tracking stream
    # persist=True maintains track IDs across frames
    # stream=True yields results per frame
    t0 = time.time()
    frame_idx = 0

    results_stream = model.track(
        source=str(video_path),
        stream=True,
        persist=True,
        tracker=args.tracker,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )

    for r in results_stream:
        frame = r.orig_img  # original frame (numpy array)

        # If no detections
        if r.boxes is None or len(r.boxes) == 0:
            writer.write(frame)
            frame_idx += 1
            continue

        # r.boxes fields: xyxy, conf, cls, id (when tracking)
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = None
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)

        # Draw + log
        for i in range(len(xyxy)):
            cls_id = int(clss[i])
            if keep_ids is not None and cls_id not in keep_ids:
                continue

            x1, y1, x2, y2 = xyxy[i].astype(int).tolist()
            conf = float(confs[i])
            track_id = int(ids[i]) if ids is not None else -1
            cls_name = r.names.get(cls_id, str(cls_id))

            # draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID {track_id} {cls_name} {conf:.2f}" if track_id != -1 else f"{cls_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # log
            if args.csv.strip():
                rows.append({
                    "frame": frame_idx,
                    "track_id": track_id,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "conf": conf,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })

        writer.write(frame)
        frame_idx += 1

    writer.release()

    elapsed = time.time() - t0
    eff_fps = frame_idx / elapsed if elapsed > 0 else 0.0
    print(f"Done. Wrote video: {out_path}")
    print(f"Processed {frame_idx} frames in {elapsed:.2f}s ({eff_fps:.2f} FPS)")

    if args.csv.strip():
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"Wrote tracks CSV: {csv_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()