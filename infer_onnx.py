"""Standalone ONNX Runtime inference for YOLO hand segmentation.

Requires only onnxruntime + numpy + opencv, no ultralytics / PyTorch.
Each post-processing step is a standalone function for easy translation.

Usage:
    uv run infer_onnx.py model.onnx image.jpg
    uv run infer_onnx.py model.onnx image.jpg --output result.png --conf 0.5
"""

import argparse
import cv2
import numpy as np
import onnxruntime as ort

IMGSZ = 224
MASK_COLOR = (0, 255, 0)
MASK_ALPHA = 0.35


# ── preprocessing ──────────────────────────────────────────────────

def preprocess(image_path: str):
    """Load BGR, resize to 224x224, normalise to [0,1]. Returns (tensor, bgr, H, W)."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    orig_h, orig_w = bgr.shape[:2]

    resized = cv2.resize(bgr, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0           # [0, 1]
    tensor = np.transpose(tensor, (2, 0, 1))              # HWC → CHW
    tensor = np.expand_dims(tensor, axis=0)               # (1, 3, 224, 224)
    return tensor, bgr, orig_h, orig_w


# ── output parsing ─────────────────────────────────────────────────

def parse_detections(output0, conf_thr):
    """Extract valid detections from end-to-end output tensor.

    output0: (1, max_det, 6 + mask_dim)
        columns: [x1, y1, x2, y2, score, class_id, m0 ... m31]
        Rows with score == 0 are padding.

    Returns (boxes_norm, scores, mask_coef) or (None, None, None) if none.
    """
    dets = output0[0]                                              # (max_det, 6 + mask_dim)
    scores = dets[:, 4]                                            # (max_det,)

    valid = scores > conf_thr
    if not valid.any():
        return None, None, None

    dets = dets[valid]
    scores = dets[:, 4]

    boxes_norm = dets[:, :4] / IMGSZ                               # [0, 1] normalised
    mask_coef = dets[:, 6:]                                        # (K, mask_dim)
    return boxes_norm, scores, mask_coef


# ── mask decoding ──────────────────────────────────────────────────

def decode_masks(proto, mask_coef, boxes_norm, ori_shape):
    """Decode instance masks from prototype + coefficients.

    proto:      (mask_dim, proto_h, proto_w)
    mask_coef:  (K, mask_dim)             – mask coefficients per detection
    boxes_norm: (K, 4) xyxy [0–1]         – boxes relative to IMGSZ
    ori_shape:  (orig_h, orig_w)

    Returns:    (K, orig_h, orig_w) boolean masks
    """
    c, mh, mw = proto.shape
    K = mask_coef.shape[0]
    ori_h, ori_w = ori_shape

    masks = np.dot(mask_coef, proto.reshape(c, -1))          # (K, mh*mw)
    masks = 1.0 / (1.0 + np.exp(-np.clip(masks, -20, 20)))  # sigmoid
    masks = masks.reshape(K, mh, mw)                         # (K, mh, mw)

    result = np.zeros((K, ori_h, ori_w), dtype=np.float32)
    for i in range(K):
        bx1 = int(boxes_norm[i, 0] * ori_w)
        by1 = int(boxes_norm[i, 1] * ori_h)
        bx2 = int(boxes_norm[i, 2] * ori_w + 1)
        by2 = int(boxes_norm[i, 3] * ori_h + 1)
        if bx2 - bx1 <= 0 or by2 - by1 <= 0:
            continue

        mx1 = int(boxes_norm[i, 0] * mw)
        my1 = int(boxes_norm[i, 1] * mh)
        mx2 = int(boxes_norm[i, 2] * mw + 1)
        my2 = int(boxes_norm[i, 3] * mh + 1)
        mx1, my1 = max(0, mx1), max(0, my1)
        mx2, my2 = min(mw, mx2), min(mh, my2)
        if mx2 - mx1 <= 0 or my2 - my1 <= 0:
            continue

        mask_crop = masks[i, my1:my2, mx1:mx2]
        mask_upsampled = cv2.resize(mask_crop, (bx2 - bx1, by2 - by1),
                                    interpolation=cv2.INTER_LINEAR)
        result[i, by1:by2, bx1:bx2] = mask_upsampled

    return result > 0.5


# ── visualisation ──────────────────────────────────────────────────

def draw_overlay(bgr, boxes, scores, masks):
    """Green semi-transparent mask overlay + bounding boxes on BGR image."""
    out = bgr.copy().astype(np.float32)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        mask = masks[i]

        out[mask] = (out[mask] * (1 - MASK_ALPHA) +
                     np.array(MASK_COLOR, dtype=np.float32) * MASK_ALPHA)

        cv2.rectangle(out, (x1, y1), (x2, y2), MASK_COLOR, 2)
        cv2.putText(out, f"hand {scores[i]:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, MASK_COLOR, 2)
    return out.astype(np.uint8)


# ── pipeline ───────────────────────────────────────────────────────

def run(model_path, image_path, conf_thr, output_path):
    tensor, bgr, orig_h, orig_w = preprocess(image_path)

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})
    output0 = outputs[0]                              # (1, max_det, 6 + mask_dim)
    output1 = outputs[1]                              # (1, mask_dim, proto_h, proto_w)

    boxes_norm, scores, mask_coef = parse_detections(output0, conf_thr)
    if boxes_norm is None:
        print("No detections above confidence threshold.")
        cv2.imwrite(output_path, bgr)
        return

    masks = decode_masks(output1[0], mask_coef, boxes_norm, (orig_h, orig_w))

    boxes_pixel = boxes_norm * np.array([orig_w, orig_h, orig_w, orig_h])
    result = draw_overlay(bgr, boxes_pixel, scores, masks)

    cv2.imwrite(output_path, result)
    print(f"Saved to {output_path}  ({len(boxes_norm)} detection(s))")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone ONNX Runtime inference for hand segmentation.")
    parser.add_argument("model", help="Path to ONNX model (.onnx)")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output", default="result.png", help="Output image path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    run(args.model, args.image, args.conf, args.output)


if __name__ == "__main__":
    main()
