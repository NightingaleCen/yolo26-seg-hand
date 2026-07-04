# HandYOLO - Hand Segmentation with YOLO26

## Reference URLs
- https://docs.ultralytics.com/datasets/segment
- https://docs.ultralytics.com/tasks/segment
- https://docs.ultralytics.com/modes/train

## Environment
- Python 3.14, managed by `uv`
- PyTorch 2.11.0+cu128, CUDA 12.8
- GPU: NVIDIA RTX 5070 Ti Laptop (12GB), compute capability 12.0
- Key deps: `ultralytics>=8.4.83`, `tensorboard>=2.21.0`, `mediapipe`, `sapiens` (git)

## Datasets

### FreiHAND (`data/FreiHAND/`)
- 130,240 training RGB / 3,960 eval RGB, all 224×224
- 32,560 mask images (view-0 only, JPEG compressed)
- Mask value range: 0~7=background, 248~255=hand (binary threshold 127)
- Split: 80/20 (seed 42) → 26,048 train / 6,512 val
- Polygon-from-mask IoU: **0.964** (200 samples)
- Single class `hand`
- YAML: `data/FreiHAND/hand_seg.yaml`

### HaGRID Classification 512p (`data/hagrid-classification-512p-no-gesture-150k/`)
- 19 gesture classes in subdirectories: call, dislike, fist, four, like, mute,
  no_gesture, ok, one, palm, peace, peace_inverted, rock, stop, stop_inverted,
  three, three2, two_up, two_up_inverted
- 153,735 total `.jpeg` images (UUID filenames, 512×512)
- `deleted_img_ids.txt` lists 1,029 excluded images (format: `{category}-{uuid}`)
- `no_gesture` class is largest (~27k), others ~7k each

### HaGRID Pseudo-Labeling Pipeline Results (`label_hagrid.py`)
- Pipeline: MediaPipe HandLandmarker → hand bbox + Sapiens2 SEG 0.4B → full-image mask → crop 224×224
- Batch processing: batch_size=8, FP16 autocast, ~6.5 it/s on RTX 5070 Ti, ~6.5h for full run
- Processed: 153,735 images
- Success (≥1 hand detected): 141,038 (91.7%)
- Excluded (no hand detected): 12,697 (8.3%)
- Total hand crops generated: 141,937
- Train/Val split (80/20, seed=42): 113,575 train / 28,362 val
- YAML: `data/HaGRID/hand_seg.yaml`

### HaGRID Output Structure
```
data/HaGRID/
├── crops/                   ← 中间产物 (split 后即可删除)
│   ├── images/              ← 224×224 .png
│   └── labels/              ← YOLO polygon .txt
├── images/{train,val}/      ← 最终训练/验证图像
├── labels/{train,val}/
├── .progress.json
└── hand_seg.yaml
```

## Data Preparation (`prepare_dataset.py`)
1. Binarized JPEG masks → PNG (threshold 127)
2. Generated YOLO polygon labels via `cv2.findContours`
3. **Critical**: mask `{N:08d}.jpg` ↔ RGB `{N:08d}.jpg` (same filename, NOT N×4)
4. 80/20 random split (seed 42) → 26,048 train / 6,512 val
5. IoU between polygon and original mask: **0.964** (200 samples)

## FreiHAND Dataset Config
- `data/FreiHAND/hand_seg.yaml` — single class `hand`
- Final structure:
```
data/FreiHAND/
├── images/{train,val}/
├── labels/{train,val}/
└── hand_seg.yaml
```
- `training/{rgb,mask}/` and `evaluation/rgb/` are raw source files (untouched)

## Model files (download manually)
| File | Source | Size | Place at |
|------|--------|------|----------|
| `sapiens2_0.4b_seg.safetensors` | [HF](https://huggingface.co/facebook/sapiens2-seg-0.4b) | ~1.6 GB | `models/sapiens2_0.4b_seg.safetensors` |
| `hand_landmarker.task` | [Kaggle](https://www.kaggle.com/models/google/hand-landmarker) or [GCS](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) | ~15 MB | `models/hand_landmarker.task` |

## Training (`train.py`)
- Model: `yolo26m-seg.pt` (23.6M params, 131.9 GFLOPs)
- Common config: imgsz=224, batch=64, amp=True, device=0, cos_lr=True,
  optimizer=auto, degrees=15 (rotation aug — hands are rotation-invariant;
  MediaPipe crop does not normalize orientation)
- Config-driven two-stage design: `STAGES` dict + `COMMON` dict, one generic
  `run_stage()`. `--stage` is **required**.

### Strategy: staged (FreiHAND → HaGRID fine-tune), not mixed
- FreiHAND (26k, clean GT, dense 161-pt polygons, clean backgrounds) → learn
  precise hand boundaries first.
- HaGRID (113k, Sapiens2 pseudo-labels, coarse 51-pt polygons, real webcam
  backgrounds; 4.36× larger) → fine-tune at low LR to adapt to deployment
  domain. `webcam.py` uses the same MediaPipe-crop pipeline as
  `label_hagrid.py`, so HaGRID is in-distribution for deployment.
- Rationale for staged over mixed: 4.36:1 ratio would let noisy coarse labels
  dominate; staged protects clean-boundary learning, then low-LR adaptation
  preserves boundary precision while learning cluttered-scene localization.
- HaGRID val is also pseudo-labeled → unreliable for early stopping, so stage 2
  runs a fixed schedule (patience > epochs) and is sanity-checked via `eval`
  on clean FreiHAND val (keep mAP50-95(M) > ~0.75).

### CLI
```bash
uv run train.py --stage freihand            # stage 1: 150ep from yolo26m-seg.pt
uv run train.py --stage freihand --resume   # resume stage 1 from last.pt
uv run train.py --stage hagrid              # stage 2: 10ep fine-tune, lr0=1e-4
uv run train.py --stage hagrid --resume     # resume stage 2 from last.pt
uv run train.py --stage eval                # val stage-2 on FreiHAND (clean GT)
```
- Resume: `YOLO(last.pt).train(resume=True)` restores weights + optimizer +
  LR scheduler + epoch; continues into the same run dir. Needs ≥1 completed
  epoch. Cosine schedule position is restored (no schedule drift).

### Run 1: FreiHAND only (legacy, 100ep)
- 100 epochs, lr0=1e-3, cos_lr=True, warmup_epochs=3, patience=20
- Output: `runs/segment/runs/hand_seg_m/`
- Best: epoch 100
  - mAP50-95(M): 0.8157
  - mAP50(M): 0.9950
  - train/seg_loss: 0.715, val/seg_loss: 0.942
- val/seg_loss still decreasing at ep100 but cosine LR had bottomed → retrain
  at 150ep (Run 2) to re-stretch the schedule.

### Run 2: FreiHAND 150ep (stage 1) — PENDING
- 150 epochs, lr0=1e-3, warmup_epochs=3, patience=20, close_mosaic=10, degrees=15
- Output: `runs/segment/runs/hand_seg_m_150ep/`
- Results: TBD

### Run 3: HaGRID fine-tune 10ep (stage 2) — PENDING
- Load Run 2 best.pt; 10 epochs, lr0=1e-4, warmup_epochs=2, close_mosaic=3
- (~43 FreiHAND-epoch-equivalent steps; low LR is the main noise/forgetting guard)
- Output: `runs/segment/runs/hand_seg_m_hagrid_ft/` → deployment model
- Results: TBD (then run `--stage eval` for boundary-precision check)

## Inference (`webcam.py`)
- Webcam → YOLO (imgsz=224, conf=0.5) → semi-transparent green mask overlay
- Bounding box + confidence label + FPS counter
- Press `q` to quit

## `label_hagrid.py` Usage
```bash
# Dry-run: re-runs all images with visualization overlay
uv run label_hagrid.py --input-dir data/hagrid_sample --max-images 100 --dry-run

# Full batch processing + auto split
uv run label_hagrid.py --input-dir data/hagrid-classification-512p-no-gesture-150k --batch-size 8

# Split-only (re-run split without re-processing)
uv run label_hagrid.py --split-only
```
