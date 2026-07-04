import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from sapiens.dense.models import init_model
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

# ── paths ────────────────────────────────────────────────────────────────
DEFAULT_INPUT = ROOT / "data" / "hagrid-classification-512p-no-gesture-150k"
DEFAULT_CROP_DIR = ROOT / "data" / "HaGRID" / "crops"
DEFAULT_SPLIT_DIR = ROOT / "data" / "HaGRID"
DEFAULT_CKPT = ROOT / "models" / "sapiens2_0.4b_seg.safetensors"
DEFAULT_MP_MODEL = ROOT / "models" / "hand_landmarker.task"

# ── constants ────────────────────────────────────────────────────────────
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
CROP_SIZE = 224
BBOX_EXPAND = 1.3
BINARY_THRESH = 127
SEG_HAND_CLASSES = {6, 15}  # Left_Hand=6, Right_Hand=15 in Sapiens2 29-class ontology
MP_MIN_CONFIDENCE = 0.5
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
SAVE_INTERVAL = 100
BATCH_SIZE = 8


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_seg_config():
    """Locate the 0.4B SEG config file shipped with the installed sapiens package."""
    import sapiens

    pkg_root = Path(sapiens.__file__).parent
    cfg = (
        pkg_root
        / "dense"
        / "configs"
        / "seg"
        / "shutterstock_goliath"
        / "sapiens2_0.4b_seg_shutterstock_goliath-1024x768.py"
    )
    if not cfg.exists():
        raise FileNotFoundError(f"Sapiens2 config not found: {cfg}")
    return str(cfg)


def stem_from_path(img_path: Path, root_dir: Path) -> str:
    """Build a stable stem_id from the image's path relative to root_dir."""
    rel = img_path.relative_to(root_dir)
    return str(rel.with_suffix("")).replace("\\", "_").replace("/", "_")


def collect_images(input_dir: Path) -> list[Path]:
    """Recursively collect all supported image files."""
    files = []
    for p in sorted(input_dir.rglob("*")):
        if p.suffix.lower() in IMG_EXTENSIONS:
            files.append(p)
    return files


def landmarks_to_bbox(normalized_landmarks, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Compute axis-aligned bbox from MediaPipe NormalizedLandmark list.

    Returns (x1, y1, x2, y2) in pixel coordinates.
    """
    xs = [lm.x * img_w for lm in normalized_landmarks]
    ys = [lm.y * img_h for lm in normalized_landmarks]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def expand_square_bbox(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int, pad_factor: float = BBOX_EXPAND):
    """Expand bbox by *pad_factor* and make it square, clamped to image bounds."""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * pad_factor / 2
    x1 = int(cx - half)
    y1 = int(cy - half)
    x2 = int(cx + half)
    y2 = int(cy + half)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    return x1, y1, x2, y2


def crop_resize(img: np.ndarray, bbox: tuple[int, int, int, int], size: int) -> np.ndarray:
    """Crop *img* by *bbox* then resize to *size*×*size*."""
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        shape = (size, size, 3) if img.ndim == 3 else (size, size)
        return np.zeros(shape, dtype=img.dtype)

    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def mask_to_yolo_polygons(binary_mask: np.ndarray, eps: float = 1.0) -> list[str]:
    """Extract contours from a binary mask and return YOLO polygon label lines."""
    binary = binary_mask.astype(np.uint8)
    h, w = binary.shape

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        cnt = cnt.squeeze()
        if cnt.ndim == 1:
            cnt = cnt.reshape(-1, 2)

        approx = cv2.approxPolyDP(cnt.astype(np.float32), eps, True).squeeze()
        if approx.ndim == 1 or len(approx) < 3:
            continue

        coords = []
        for pt in approx:
            coords.append(round(pt[0] / w, 6))
            coords.append(round(pt[1] / h, 6))
        lines.append(f"0 {' '.join(map(str, coords))}")
    return lines


def landmark_convex_hull_mask(normalized_landmarks, img_w: int, img_h: int) -> np.ndarray:
    """Fallback: create a binary mask from the convex hull of MediaPipe hand landmarks."""
    hull_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    pts = np.array(
        [[int(lm.x * img_w), int(lm.y * img_h)] for lm in normalized_landmarks],
        dtype=np.int32,
    )
    hull = cv2.convexHull(pts)
    cv2.fillPoly(hull_mask, [hull], 255)
    return hull_mask


def draw_hand_overlays(orig_img, bbox, crop_img, mask_crop, hand_idx, hand_mask_full=None):
    """Create a composite visualization image for dry-run inspection."""
    x1, y1, x2, y2 = bbox

    vis = orig_img.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(vis, f"H{hand_idx}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if hand_mask_full is not None:
        overlay = np.zeros_like(orig_img)
        overlay[hand_mask_full > 0] = (0, 255, 0)
        vis = cv2.addWeighted(vis, 1.0, overlay, 0.3, 0)

    crop_vis = crop_img.copy()
    if mask_crop is not None and mask_crop.size > 0:
        mask_overlay = np.zeros_like(crop_vis)
        mask_overlay[mask_crop > 0] = (0, 255, 0)
        crop_vis = cv2.addWeighted(crop_vis, 1.0, mask_overlay, 0.4, 0)
        cv2.putText(
            crop_vis,
            f"mask area={(mask_crop > 0).sum()}",
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )

    # stack vertically: original image with bbox on top, crop+mask below
    h_vis = vis.shape[0]
    h_crop = crop_vis.shape[0]
    combined = np.zeros((h_vis + h_crop, vis.shape[1], 3), dtype=np.uint8)
    combined[:h_vis, :vis.shape[1]] = vis
    combined[h_vis : h_vis + h_crop, :crop_vis.shape[1]] = crop_vis
    return combined


# ═══════════════════════════════════════════════════════════════════════════
# core pipeline
# ═══════════════════════════════════════════════════════════════════════════

class HagridLabeler:
    def __init__(self, checkpoint: str, mp_model: str, batch_size: int = BATCH_SIZE, device: str = "cuda:0"):
        self.device = device
        self.batch_size = batch_size

        print("Loading Sapiens2 SEG model (FP16) …", file=sys.stderr)
        cfg = get_seg_config()
        self.model = init_model(cfg, checkpoint, device=device)
        self.model.half()
        print(f"  Batch size: {batch_size}", file=sys.stderr)

        print("Loading MediaPipe HandLandmarker …", file=sys.stderr)
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=mp_model),
            running_mode=RunningMode.IMAGE,
            num_hands=10,
            min_hand_detection_confidence=MP_MIN_CONFIDENCE,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def _sapiens2_segment_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Run Sapiens2 SEG on a batch of images (FP16 autocast).

        Returns a list of binary hand masks at each image's original resolution.
        """
        if not images:
            return []

        tensors = []
        for img in images:
            data = self.model.pipeline(dict(img=img))
            data = self.model.data_preprocessor(data)
            tensors.append(data["inputs"])

        batch = torch.cat(tensors, dim=0)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                seg_logits = self.model(batch)

        masks = []
        for i, img in enumerate(images):
            logits = seg_logits[i : i + 1].float()
            logits = F.interpolate(logits, size=img.shape[:2], mode="bilinear")
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            mask = np.isin(pred, list(SEG_HAND_CLASSES)).astype(np.uint8) * 255
            masks.append(mask)

        return masks

    def _detect_hands(self, image: np.ndarray):
        """Run MediaPipe hand detection. Returns list of NormalizedLandmark lists."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        return result.hand_landmarks or []

    def _crop_hands(self, image: np.ndarray, hand_mask_full: np.ndarray, landmarks_list) -> list[dict]:
        """Extract per-hand crops + YOLO labels from a full-image mask."""
        h, w = image.shape[:2]
        crops = []
        for hand_idx, landmarks in enumerate(landmarks_list):
            bbox_raw = landmarks_to_bbox(landmarks, w, h)
            bbox = expand_square_bbox(*bbox_raw, w, h)

            mask_cropped = crop_resize(hand_mask_full, bbox, CROP_SIZE)
            mask_binary = (mask_cropped > BINARY_THRESH).astype(np.uint8)

            if mask_binary.sum() < 100:
                hull = landmark_convex_hull_mask(landmarks, w, h)
                mask_cropped = crop_resize(hull, bbox, CROP_SIZE)
                mask_binary = (mask_cropped > 0).astype(np.uint8) * 255

            img_cropped = crop_resize(image, bbox, CROP_SIZE)

            lines = mask_to_yolo_polygons(mask_binary, eps=1.0)
            if not lines:
                continue

            crops.append(
                {
                    "img": img_cropped,
                    "mask": mask_binary,
                    "lines": lines,
                    "bbox": bbox,
                    "hand_idx": hand_idx,
                }
            )
        return crops

    def close(self):
        self.landmarker.close()


# ═══════════════════════════════════════════════════════════════════════════
# batch processing with resume
# ═══════════════════════════════════════════════════════════════════════════

def run_batch(
    input_dir: Path,
    crop_dir: Path,
    max_images: int = 0,
    dry_run: bool = False,
    batch_size: int = BATCH_SIZE,
    checkpoint_path: str = "",
    mp_model_path: str = "",
):
    images_out = crop_dir / "images"
    labels_out = crop_dir / "labels"
    vis_out = crop_dir / "vis"
    progress_file = crop_dir.parent / ".progress.json"

    for d in [images_out, labels_out]:
        d.mkdir(parents=True, exist_ok=True)
    if dry_run:
        vis_out.mkdir(parents=True, exist_ok=True)

    progress: dict = {}
    if progress_file.exists():
        progress = json.loads(progress_file.read_text(encoding="utf-8"))

    all_images = collect_images(input_dir)
    if max_images > 0:
        all_images = all_images[:max_images]

    processed = progress.get("processed", {})
    already = set(processed.keys())

    todo = []
    for p in all_images:
        sid = stem_from_path(p, input_dir)
        skip = sid in already
        if not skip and not dry_run and (images_out / f"{sid}_0.png").exists():
            skip = True
        if not skip:
            todo.append(p)

    total_todo = len(todo)
    print(f"Total: {len(all_images)}  already: {len(already)}  to process: {total_todo}")

    if total_todo == 0:
        print("Nothing to do.")
        return

    labeller = HagridLabeler(
        checkpoint=str(checkpoint_path),
        mp_model=str(mp_model_path),
        batch_size=batch_size,
    )

    stats = {"success": 0, "no_hand": 0, "total_hands": 0}
    t0 = time.perf_counter()
    pbar = tqdm(total=total_todo, desc="Labeling", unit="img")

    # ── process in mini-batches: MP first, then GPU batch ──
    for batch_start in range(0, total_todo, batch_size):
        batch_slice = todo[batch_start : batch_start + batch_size]

        # Step 1: load images + MediaPipe detection (CPU)
        batch_data: list[dict] = []
        images_for_sapiens: list[np.ndarray] = []

        for img_path in batch_slice:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            hands = labeller._detect_hands(image)
            entry = {"path": img_path, "image": image, "hands": hands}
            batch_data.append(entry)
            if hands:
                images_for_sapiens.append(image)

        # Step 2: Sapiens2 batch inference (GPU, one call)
        masks_batch: list[np.ndarray] = []
        if images_for_sapiens:
            try:
                masks_batch = labeller._sapiens2_segment_batch(images_for_sapiens)
            except Exception as e:
                print(f"  [WARN] Sapiens2 batch failed: {e}", file=sys.stderr)
                masks_batch = []

        # Step 3: per-image crop + save
        mask_idx = 0
        for entry in batch_data:
            img_path = entry["path"]
            image = entry["image"]
            hands = entry["hands"]
            sid = stem_from_path(img_path, input_dir)

            if not hands:
                stats["no_hand"] += 1
                processed[sid] = 0
                pbar.update(1)
                continue

            if mask_idx >= len(masks_batch):
                pbar.update(1)
                continue

            hand_mask_full = masks_batch[mask_idx]
            mask_idx += 1

            crops = labeller._crop_hands(image, hand_mask_full, hands)
            if not crops:
                stats["no_hand"] += 1
                processed[sid] = 0
                pbar.update(1)
                continue

            stats["success"] += 1
            stats["total_hands"] += len(crops)
            processed[sid] = len(crops)

            for crop in crops:
                hidx = crop["hand_idx"]
                out_name = f"{sid}_{hidx}"

                cv2.imwrite(str(images_out / f"{out_name}.png"), crop["img"])
                label_path = labels_out / f"{out_name}.txt"
                label_path.write_text("\n".join(crop["lines"]) + "\n", encoding="utf-8")

                if dry_run and hand_mask_full is not None:
                    orig = cv2.imread(str(img_path))
                    vis = draw_hand_overlays(
                        orig, crop["bbox"], crop["img"], crop["mask"], hidx, hand_mask_full
                    )
                    cv2.imwrite(str(vis_out / f"{out_name}_overlay.png"), vis)

            pbar.update(1)

        # periodic flush
        if not dry_run and stats["success"] > 0 and stats["success"] % SAVE_INTERVAL == 0:
            progress_file.write_text(
                json.dumps({"processed": processed, "stats": stats}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    pbar.close()
    labeller.close()

    elapsed = time.perf_counter() - t0
    it_per_sec = total_todo / elapsed if elapsed > 0 else 0
    if not dry_run:
        progress_file.write_text(
            json.dumps({"processed": processed, "stats": stats}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(
        f"\nDone. success={stats['success']}, no_hand={stats['no_hand']}, "
        f"total_hands={stats['total_hands']}, "
        f"elapsed={elapsed:.1f}s, {it_per_sec:.1f} it/s"
    )


# ═══════════════════════════════════════════════════════════════════════════
# train/val split
# ═══════════════════════════════════════════════════════════════════════════

def run_split(
    crop_dir: Path,
    split_dir: Path,
    train_ratio: float = TRAIN_RATIO,
    seed: int = RANDOM_SEED,
):
    images_src = crop_dir / "images"
    labels_src = crop_dir / "labels"

    if not images_src.exists():
        raise FileNotFoundError(f"Crop images dir not found: {images_src}")

    images_train = split_dir / "images" / "train"
    images_val = split_dir / "images" / "val"
    labels_train = split_dir / "labels" / "train"
    labels_val = split_dir / "labels" / "val"

    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    # collect unique stem_ids (stem before trailing _\d+)
    stem_ids: set[str] = set()
    for f in images_src.glob("*.png"):
        stem = f.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            stem_ids.add(parts[0])
        else:
            stem_ids.add(stem)

    id_list = sorted(stem_ids)
    random.seed(seed)
    random.shuffle(id_list)
    split_n = int(len(id_list) * train_ratio)
    train_set = set(id_list[:split_n])
    val_set = set(id_list[split_n:])

    print(f"Total source images (stem): {len(id_list)}, train={len(train_set)}, val={len(val_set)}")

    copied = 0
    for f in sorted(images_src.glob("*.png")):
        stem = f.stem.rsplit("_", 1)
        stem_id = stem[0] if (len(stem) == 2 and stem[1].isdigit()) else f.stem
        label_f = labels_src / f"{f.stem}.txt"

        if stem_id in train_set:
            dst_img, dst_lbl = images_train / f.name, labels_train / f"{f.stem}.txt"
        else:
            dst_img, dst_lbl = images_val / f.name, labels_val / f"{f.stem}.txt"

        shutil.copy2(str(f), str(dst_img))
        if label_f.exists():
            shutil.copy2(str(label_f), str(dst_lbl))
        copied += 1

    print(f"Copied {copied} crop images.")


def create_yaml(split_dir: Path):
    """Write hand_seg.yaml for the HaGRID dataset."""
    content = (
        f"# Hand segmentation dataset (HaGRID, Sapiens2 pseudo-labeled)\n"
        f"path: {split_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names:\n"
        f"  0: hand\n"
    )
    yaml_path = split_dir / "hand_seg.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    print(f"Written: {yaml_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pseudo-label HaGRID with MediaPipe + Sapiens2")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT, help="Raw images root")
    parser.add_argument("--crop-dir", type=Path, default=DEFAULT_CROP_DIR, help="Stage-1 crop output")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR, help="Stage-2 split target")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT), help="Sapiens2 seg checkpoint")
    parser.add_argument("--mp-model", type=str, default=str(DEFAULT_MP_MODEL), help="MediaPipe hand_landmarker.task")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Sapiens2 batch size")
    parser.add_argument("--max-images", type=int, default=0, help="Limit images (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Run with visualization output")
    parser.add_argument("--split-only", action="store_true", help="Only run train/val split (skip labeling)")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for split")
    args = parser.parse_args()

    if args.split_only:
        run_split(args.crop_dir, args.split_dir, args.train_ratio, args.seed)
        create_yaml(args.split_dir)
    else:
        run_batch(
            args.input_dir,
            args.crop_dir,
            max_images=args.max_images,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            checkpoint_path=args.checkpoint,
            mp_model_path=args.mp_model,
        )
        if not args.dry_run:
            print("\nRunning train/val split …")
            run_split(args.crop_dir, args.split_dir, args.train_ratio, args.seed)
            create_yaml(args.split_dir)


if __name__ == "__main__":
    main()
