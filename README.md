# yolo26-seg-hand

Real-time hand segmentation with [YOLO26m-seg](https://github.com/ultralytics/ultralytics), designed to pair with
MediaPipe HandLandmarker's bounding box for per-hand 224×224 crop inference. Performs well on webcam photos.

## Model weights

Pre-trained weights are on Hugging Face:
**[NightingaleCen/YOLO26m-seg-hand](https://huggingface.co/NightingaleCen/YOLO26m-seg-hand)**

| File | Size |
|---|---|
| `yolo26m-seg-hand.pt` | ~52 MB |
| `yolo26m-seg-hand.onnx` | ~90 MB |

## Training

Two-stage training on a single RTX 5070 Ti Laptop:

| Stage | Dataset | Epochs | LR | mAP50-95(M) |
|---|---|---|---|---|
| 1 | FreiHAND (26k, clean GT) | 150 | 1e‑3 | 0.831 |
| 2 | HaGRID subset (113k, Sapiens2 pseudo-labels) | 10 | 1e‑4 | 0.902† |

† HaGRID val is also pseudo-labeled

See [`train.py`](train.py) for the config-driven two-stage training script.

## Demo

```bash
uv run webcam.py                # MediaPipe crop + YOLO
```

Press `q` to quit.

## ONNX export

```bash
uv run export_onnx.py              # FP32, ~90 MB
uv run export_onnx.py --fp16       # FP16, ~45 MB
```

Standalone inference:

```bash
uv run infer_onnx.py yolo26m-seg-hand.onnx image.jpg --output out.png
```

Every post‑processing step in [`infer_onnx.py`](infer_onnx.py) is a standalone function with shape annotations,
designed to be translatable to C++ / C# / Rust / JS.

## Project structure

```
├── train.py               # two-stage training (--stage freihand|hagrid|eval)
├── webcam.py              # real-time demo
├── label_hagrid.py        # MediaPipe + Sapiens2 pseudo-labeling pipeline
├── prepare_dataset.py     # FreiHAND mask → YOLO polygon labels
├── export_onnx.py         # .pt → .onnx export
├── infer_onnx.py          # standalone ONNX Runtime inference
├── data/                  # datasets (gitignored)
│   ├── FreiHAND/          
│   └── HaGRID/            
├── models/                # download sapiens2*.safetensors, hand_landmarker.task
├── runs/segment/runs/     # training outputs (gitignored)
└── pyproject.toml
```

## Requirements

- Python 3.14, managed by `uv`
- PyTorch 2.11.0+cu128, CUDA 12.8
- `ultralytics >= 8.4.83`
- `mediapipe`, `sapiens`, `onnxruntime` (optional, for export / inference)

```bash
uv sync
```

## Datasets

- [FreiHAND](https://lmb.informatik.uni-freiburg.de/projects/freihand/) — 130k training RGB images with ground-truth masks (view‑0 only, 224×224)
- [HaGRID](https://github.com/hukenovs/hagrid) — subset with 153k hand-gesture images, pseudo‑labeled with Sapiens2 SEG 0.4B via [`label_hagrid.py`](label_hagrid.py)

## License

AGPL-3.0 — this model is a fine-tuned derivative of [Ultralytics YOLO26m-seg](https://github.com/ultralytics/ultralytics).
