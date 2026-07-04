import time
from pathlib import Path
import argparse

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = "runs/segment/runs/hand_seg_m_hagrid_ft/weights/best.pt"
MP_MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
IMGSZ = 224
CONF = 0.5
MASK_COLOR = (0, 255, 0)
MASK_ALPHA = 0.35
FPS_SMOOTH = 0.9
BBOX_EXPAND = 1.3
MP_MIN_CONFIDENCE = 0.5


def landmarks_to_bbox(normalized_landmarks, img_w: int, img_h: int):
    xs = [lm.x * img_w for lm in normalized_landmarks]
    ys = [lm.y * img_h for lm in normalized_landmarks]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def expand_square_bbox(x1, y1, x2, y2, img_w, img_h, pad_factor=BBOX_EXPAND):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * pad_factor / 2
    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(img_w, int(cx + half))
    y2 = min(img_h, int(cy + half))
    return x1, y1, x2, y2


def map_coords(points, x1, y1, crop_w, crop_h):
    return points * np.array([crop_w / IMGSZ, crop_h / IMGSZ]) + np.array([x1, y1])


def _run_full_frame(cap, model):
    fps_display = 0.0
    print("Full-frame mode (no MediaPipe crop, press q to quit)")

    while True:
        t_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        x_scale = w / IMGSZ
        y_scale = h / IMGSZ

        small = cv2.resize(frame, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
        results = model(small, imgsz=IMGSZ, conf=CONF, verbose=False)

        if results[0].masks is not None:
            overlay = frame.copy()
            xy = results[0].masks.xy
            confs = results[0].boxes.conf.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for i, polygon in enumerate(xy):
                pts = (polygon * np.array([x_scale, y_scale])).astype(np.int32)
                cv2.fillPoly(overlay, [pts], MASK_COLOR)
                cv2.polylines(frame, [pts], True, MASK_COLOR, 2)

                if i < len(boxes):
                    b = (boxes[i] * np.array([x_scale, y_scale, x_scale, y_scale])).astype(int)
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), MASK_COLOR, 2)
                    cv2.putText(frame, f"hand {confs[i]:.2f}", (b[0], b[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, MASK_COLOR, 2)

            cv2.addWeighted(overlay, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0, frame)

        dt = time.perf_counter() - t_start
        if fps_display == 0.0:
            fps_display = 1.0 / dt if dt > 0 else 0
        else:
            fps_display = FPS_SMOOTH * fps_display + (1 - FPS_SMOOTH) * (1.0 / dt) if dt > 0 else fps_display

        cv2.putText(frame, f"FPS: {fps_display:.1f} (full-frame)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Hand Segmentation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main():
    parser = argparse.ArgumentParser(description="Hand segmentation webcam demo")
    parser.add_argument("--full-frame", action="store_true",
                        help="Skip MediaPipe hand cropping; run YOLO on full 224x224 frame")
    args = parser.parse_args()

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if args.full_frame:
        _run_full_frame(cap, model)
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return

    print("Loading MediaPipe HandLandmarker...")
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MP_MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=10,
        min_hand_detection_confidence=MP_MIN_CONFIDENCE,
    )
    with HandLandmarker.create_from_options(options) as landmarker:
        fps_display = 0.0
        label_font = cv2.FONT_HERSHEY_SIMPLEX

        print("Running webcam inference (q to quit)")

        while True:
            t_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                overlay = frame.copy()
                crop_previews = []

                for landmarks in result.hand_landmarks:
                    raw_bbox = landmarks_to_bbox(landmarks, w, h)
                    x1, y1, x2, y2 = expand_square_bbox(*raw_bbox, w, h)
                    crop_w, crop_h = x2 - x1, y2 - y1

                    if crop_w <= 0 or crop_h <= 0:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    crop = cv2.resize(crop, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
                    crop_previews.append(crop)

                    results = model(crop, imgsz=IMGSZ, conf=CONF, verbose=False)

                    if results[0].masks is None:
                        continue

                    xy = results[0].masks.xy
                    confs = results[0].boxes.conf.cpu().numpy()
                    boxes = results[0].boxes.xyxy.cpu().numpy()

                    for i, polygon in enumerate(xy):
                        pts = map_coords(polygon, x1, y1, crop_w, crop_h).astype(np.int32)

                        cv2.fillPoly(overlay, [pts], MASK_COLOR)
                        cv2.polylines(frame, [pts], True, MASK_COLOR, 2)

                        if i < len(boxes):
                            bx1, by1, bx2, by2 = map_coords(
                                boxes[i].reshape(-1, 2), x1, y1, crop_w, crop_h
                            ).ravel().astype(int)
                            cv2.rectangle(frame, (bx1, by1), (bx2, by2), MASK_COLOR, 2)
                            label = f"hand {confs[i]:.2f}"
                            cv2.putText(frame, label, (bx1, by1 - 5),
                                        label_font, 0.5, MASK_COLOR, 2)

                cv2.addWeighted(overlay, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0, frame)

                preview_sz = 112
                per_row = max(1, (w - 20) // (preview_sz + 4))
                for idx, prev in enumerate(crop_previews):
                    thumb = cv2.resize(prev, (preview_sz, preview_sz))
                    row = idx // per_row
                    col = idx % per_row
                    px = w - (col + 1) * (preview_sz + 4) - 4
                    py = 8 + row * (preview_sz + 4)
                    if py + preview_sz <= h and px >= 0:
                        cv2.rectangle(frame, (px - 1, py - 1),
                                      (px + preview_sz, py + preview_sz), (255, 255, 255), 1)
                        frame[py:py + preview_sz, px:px + preview_sz] = thumb

            dt = time.perf_counter() - t_start
            if fps_display == 0.0:
                fps_display = 1.0 / dt if dt > 0 else 0
            else:
                fps_display = FPS_SMOOTH * fps_display + (1 - FPS_SMOOTH) * (1.0 / dt) if dt > 0 else fps_display

            cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 25),
                        label_font, 0.7, (0, 255, 255), 2)

            cv2.imshow("Hand Segmentation", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    main()
