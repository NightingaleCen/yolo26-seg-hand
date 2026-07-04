import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"E:\Projects\HandYOLO")
DATA_DIR = ROOT / "data" / "FreiHAND"
RGB_DIR = DATA_DIR / "training" / "rgb"
MASK_DIR = DATA_DIR / "training" / "mask"

MASK_BINARY_DIR = DATA_DIR / "masks_binary"
LABELS_ALL_DIR = DATA_DIR / "labels_all"
IMAGES_TRAIN_DIR = DATA_DIR / "images" / "train"
IMAGES_VAL_DIR = DATA_DIR / "images" / "val"
LABELS_TRAIN_DIR = DATA_DIR / "labels" / "train"
LABELS_VAL_DIR = DATA_DIR / "labels" / "val"

TRAIN_RATIO = 0.80
RANDOM_SEED = 42


def step1_binarize_masks():
    """Convert JPEG masks to binary PNG masks (threshold 127)."""
    print("=" * 60)
    print("Step 1: Binarizing masks (JPEG -> binary PNG)")
    print("=" * 60)

    MASK_BINARY_DIR.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(MASK_DIR.glob("*.jpg"))
    total = len(mask_files)
    print(f"  Found {total} mask images")

    for i, mask_path in enumerate(mask_files):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  WARNING: failed to read {mask_path.name}")
            continue

        binary = (mask > 127).astype(np.uint8)
        out_path = MASK_BINARY_DIR / f"{mask_path.stem}.png"
        cv2.imwrite(str(out_path), binary)

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1}/{total}...")

    print(f"  Done. Binary masks saved to: {MASK_BINARY_DIR}")
    print(f"  Count: {len(list(MASK_BINARY_DIR.glob('*.png')))}")


def generate_label_from_mask(mask_path, output_dir, img_width, img_height):
    """Generate YOLO polygon label from a single binary mask."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"  WARNING: failed to read {mask_path.name}")
        return False
    if mask.ndim != 2:
        print(f"  WARNING: unexpected ndim={mask.ndim} for {mask_path.name}")
        return False

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []
    for contour in contours:
        if len(contour) >= 3:
            contour = contour.squeeze()
            if contour.ndim == 1:
                contour = contour.reshape(-1, 2)
            coords = []
            for point in contour:
                coords.append(round(point[0] / img_width, 6))
                coords.append(round(point[1] / img_height, 6))
            lines.append(f"0 {' '.join(map(str, coords))}")

    output_path = Path(output_dir) / f"{mask_path.stem}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")
    return True


def step2_generate_labels():
    """Generate YOLO-format polygon labels from binary masks."""
    print()
    print("=" * 60)
    print("Step 2: Generating YOLO polygon labels")
    print("=" * 60)

    LABELS_ALL_DIR.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(MASK_BINARY_DIR.glob("*.png"))
    total = len(mask_files)
    failed = 0
    empty = 0

    for i, mask_path in enumerate(mask_files):
        ok = generate_label_from_mask(mask_path, str(LABELS_ALL_DIR), 224, 224)
        if not ok:
            failed += 1
        else:
            label_path = LABELS_ALL_DIR / f"{mask_path.stem}.txt"
            if label_path.stat().st_size == 0:
                empty += 1

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1}/{total}...")

    label_count = len(list(LABELS_ALL_DIR.glob("*.txt")))
    print(f"  Done. Labels saved to: {LABELS_ALL_DIR}")
    print(f"  Count: {label_count}, Failed: {failed}, Empty: {empty}")


def step3_split_and_organize():
    """Split data 80/20 and copy images + labels to final structure."""
    print()
    print("=" * 60)
    print("Step 3: Splitting data and organizing directories")
    print("=" * 60)

    for d in [IMAGES_TRAIN_DIR, IMAGES_VAL_DIR, LABELS_TRAIN_DIR, LABELS_VAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    label_files = sorted(LABELS_ALL_DIR.glob("*.txt"))
    id_list = [int(p.stem) for p in label_files]
    total = len(id_list)
    print(f"  Total samples: {total}")

    random.seed(RANDOM_SEED)
    random.shuffle(id_list)

    split_idx = int(total * TRAIN_RATIO)
    train_ids = set(id_list[:split_idx])
    val_ids = set(id_list[split_idx:])
    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}")

    print("  Copying images and labels...")
    for i, sample_id in enumerate(sorted(train_ids | val_ids)):
        rgb_filename = f"{sample_id:08d}.jpg"
        src_rgb = RGB_DIR / rgb_filename
        label_filename = f"{sample_id:08d}.txt"
        src_label = LABELS_ALL_DIR / label_filename

        if sample_id in train_ids:
            dst_rgb = IMAGES_TRAIN_DIR / f"{sample_id:08d}.jpg"
            dst_label = LABELS_TRAIN_DIR / label_filename
        else:
            dst_rgb = IMAGES_VAL_DIR / f"{sample_id:08d}.jpg"
            dst_label = LABELS_VAL_DIR / label_filename

        if not src_rgb.exists():
            print(f"  WARNING: missing RGB {src_rgb}")
            continue
        if not src_label.exists():
            print(f"  WARNING: missing label {src_label}")
            continue

        shutil.copy2(str(src_rgb), str(dst_rgb))
        shutil.copy2(str(src_label), str(dst_label))

        if (i + 1) % 5000 == 0:
            print(f"  Copied {i+1}/{total}...")

    print(f"  Done.")
    print(f"  Train images: {len(list(IMAGES_TRAIN_DIR.glob('*.jpg')))}")
    print(f"  Val   images: {len(list(IMAGES_VAL_DIR.glob('*.jpg')))}")
    print(f"  Train labels: {len(list(LABELS_TRAIN_DIR.glob('*.txt')))}")
    print(f"  Val   labels: {len(list(LABELS_VAL_DIR.glob('*.txt')))}")


def step4_create_yaml():
    """Create the dataset YAML configuration file."""
    print()
    print("=" * 60)
    print("Step 4: Creating hand_seg.yaml")
    print("=" * 60)

    yaml_content = f"""# Hand segmentation dataset (FreiHAND, view-0 only)
path: {DATA_DIR.as_posix()}
train: images/train
val: images/val

names:
  0: hand
"""

    yaml_path = DATA_DIR / "hand_seg.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"  Saved to: {yaml_path}")
    print(f"  Content:")
    print(yaml_content)


def main():
    print("=" * 60)
    print("Preparing FreiHAND dataset for YOLO segmentation")
    print("=" * 60)
    print()

    step1_binarize_masks()
    step2_generate_labels()
    step3_split_and_organize()
    step4_create_yaml()

    print()
    print("=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print()
    print("Final structure:")
    print(f"  {DATA_DIR / 'images' / 'train'}  ← train RGB images")
    print(f"  {DATA_DIR / 'images' / 'val'}    ← val RGB images")
    print(f"  {DATA_DIR / 'labels' / 'train'}  ← train polygon labels")
    print(f"  {DATA_DIR / 'labels' / 'val'}    ← val polygon labels")
    print(f"  {DATA_DIR / 'hand_seg.yaml'}     ← dataset config")
    print()
    print("To verify, check a few label files against their images.")


if __name__ == "__main__":
    main()
