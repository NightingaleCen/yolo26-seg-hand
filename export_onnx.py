"""Export trained YOLO segmentation model to ONNX format."""

import argparse
from ultralytics import YOLO

MODEL_PATH = "runs/segment/runs/hand_seg_m_hagrid_ft/weights/best.pt"


def main():
    parser = argparse.ArgumentParser(description="Export hand seg model to ONNX.")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to .pt weights")
    parser.add_argument("--fp16", action="store_true", help="FP16 precision (smaller, faster on GPU)")
    args = parser.parse_args()

    kwargs = dict(format="onnx", imgsz=224, simplify=True, opset=17)
    if args.fp16:
        kwargs["quantize"] = 16

    model = YOLO(args.model)
    path = model.export(**kwargs)
    print(f"Exported to: {path}")


if __name__ == "__main__":
    main()
