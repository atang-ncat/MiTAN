#!/usr/bin/env python3
"""
HybridNets ONNX Inference — Standalone (no PyTorch required)

Usage:
    python3 hybridnets_inference.py --image path/to/image.jpg
    python3 hybridnets_inference.py --image_dir path/to/folder --output results/

Dependencies (Jetson AGX Orin):
    pip install onnxruntime-gpu opencv-python numpy pyyaml
    # or for TensorRT acceleration:
    pip install onnxruntime-gpu  (comes with TensorRT EP on Jetson)
"""

import argparse
import os
import time
import cv2
import numpy as np
import yaml
from glob import glob


# ─── Configuration ───────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "weights", "hybridnets_384x640.onnx")
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "config", "our_dataset.yml")

# Detection class colors (BGR for OpenCV)
CLASS_COLORS = {
    'person':              (0, 255, 0),    # green
    'robot':               (255, 0, 0),    # blue
    'traffic_light_green': (0, 255, 128),  # bright green
    'traffic_light_off':   (128, 128, 128),# gray
    'traffic_light_red':   (0, 0, 255),    # red
}

# Segmentation overlay colors (BGR)
SEG_COLORS = {
    'road': (255, 0, 255),  # magenta/pink
    'lane': (0, 255, 0),    # green
}

INPUT_H = 384
INPUT_W = 640


# ─── Preprocessing ──────────────────────────────────────────────────────────

def letterbox(img, new_shape=(INPUT_H, INPUT_W), color=(114, 114, 114), stride=128):
    """Resize and pad image to exact shape, maintaining aspect ratio."""
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = int(round(w * r)), int(round(h * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def preprocess(img_bgr, mean, std):
    """Preprocess image for ONNX inference."""
    img_lb, ratio, (dw, dh) = letterbox(img_bgr, (INPUT_H, INPUT_W), stride=128)
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_float = (img_float - np.array(mean)) / np.array(std)
    img_chw = np.transpose(img_float, (2, 0, 1))  # HWC -> CHW
    img_batch = np.expand_dims(img_chw, axis=0)    # add batch dim
    return img_batch.astype(np.float32), ratio, dw, dh


# ─── Post-processing ────────────────────────────────────────────────────────

def decode_detections(regression, classification, anchors, conf_thresh=0.25,
                      iou_thresh=0.45):
    """Decode raw detection outputs into bounding boxes + labels + scores."""
    # classification shape: (1, num_anchors, num_classes) — already sigmoid-ed by ONNX model
    scores = classification[0]  # no sigmoid needed, model applies it in onnx_export mode
    reg = regression[0]

    # Get max class score for each anchor
    max_scores = np.max(scores, axis=1)
    max_classes = np.argmax(scores, axis=1)

    # Filter by confidence
    mask = max_scores > conf_thresh
    if not np.any(mask):
        return [], [], []

    filtered_scores = max_scores[mask]
    filtered_classes = max_classes[mask]
    filtered_reg = reg[mask]
    filtered_anchors = anchors[mask]

    # Decode boxes from regression + anchors
    # reg format: (dx, dy, dw, dh) relative to anchor
    ax, ay, aw, ah = (filtered_anchors[:, 0], filtered_anchors[:, 1],
                      filtered_anchors[:, 2], filtered_anchors[:, 3])
    dx, dy, dw, dh = (filtered_reg[:, 0], filtered_reg[:, 1],
                      filtered_reg[:, 2], filtered_reg[:, 3])

    cx = ax + dx * aw
    cy = ay + dy * ah
    w = aw * np.exp(dw)
    h = ah * np.exp(dh)

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # NMS per class
    final_boxes, final_scores, final_classes = [], [], []
    for cls_id in np.unique(filtered_classes):
        cls_mask = filtered_classes == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = filtered_scores[cls_mask]

        # Simple NMS
        order = cls_scores.argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            xx1 = np.maximum(cls_boxes[i, 0], cls_boxes[order[1:], 0])
            yy1 = np.maximum(cls_boxes[i, 1], cls_boxes[order[1:], 1])
            xx2 = np.minimum(cls_boxes[i, 2], cls_boxes[order[1:], 2])
            yy2 = np.minimum(cls_boxes[i, 3], cls_boxes[order[1:], 3])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area_i = (cls_boxes[i, 2] - cls_boxes[i, 0]) * (cls_boxes[i, 3] - cls_boxes[i, 1])
            area_j = ((cls_boxes[order[1:], 2] - cls_boxes[order[1:], 0]) *
                      (cls_boxes[order[1:], 3] - cls_boxes[order[1:], 1]))
            iou = inter / (area_i + area_j - inter + 1e-6)
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]

        for k in keep:
            final_boxes.append(cls_boxes[k])
            final_scores.append(cls_scores[k])
            final_classes.append(cls_id)

    return final_boxes, final_scores, final_classes


def generate_anchors(image_h, image_w, scales_str, ratios_str, pyramid_levels=None):
    """Generate EfficientDet-style anchors for the given image size."""
    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    scales = np.array(eval(scales_str))
    ratios = np.array(eval(ratios_str))

    all_anchors = np.zeros((0, 4), dtype=np.float32)

    for level in pyramid_levels:
        stride = 2 ** level
        feature_h = int(np.ceil(image_h / stride))
        feature_w = int(np.ceil(image_w / stride))

        # Base anchors for this level
        base_anchors = []
        for s in scales:
            for r_h, r_w in ratios:
                anchor_h = stride * s * r_h
                anchor_w = stride * s * r_w
                base_anchors.append([-anchor_w / 2, -anchor_h / 2,
                                     anchor_w / 2,  anchor_h / 2])
        base_anchors = np.array(base_anchors, dtype=np.float32)

        # Shift anchors to each grid position
        shift_x = np.arange(0, feature_w) * stride + stride / 2
        shift_y = np.arange(0, feature_h) * stride + stride / 2
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.stack([shift_x.ravel(), shift_y.ravel(),
                          shift_x.ravel(), shift_y.ravel()], axis=1)

        level_anchors = (shifts[:, np.newaxis, :] + base_anchors[np.newaxis, :, :])
        level_anchors = level_anchors.reshape(-1, 4)
        all_anchors = np.concatenate([all_anchors, level_anchors], axis=0)

    # Convert from (x1, y1, x2, y2) to (cx, cy, w, h) for decoding
    cx = (all_anchors[:, 0] + all_anchors[:, 2]) / 2
    cy = (all_anchors[:, 1] + all_anchors[:, 3]) / 2
    w = all_anchors[:, 2] - all_anchors[:, 0]
    h = all_anchors[:, 3] - all_anchors[:, 1]
    anchors_cxcywh = np.stack([cx, cy, w, h], axis=1)

    return anchors_cxcywh


def decode_segmentation(seg_output, num_classes=3):
    """Decode segmentation output to class masks.

    Returns:
        road_mask: binary mask for drivable area
        lane_mask: binary mask for lane lines
    """
    # seg_output shape: (1, num_classes, H, W) — multiclass mode
    seg = seg_output[0]  # remove batch dim
    class_ids = np.argmax(seg, axis=0)  # (H, W) — per-pixel class

    # Class 0 = background, Class 1 = road, Class 2 = lane
    road_mask = (class_ids == 1).astype(np.uint8)
    lane_mask = (class_ids == 2).astype(np.uint8)

    return road_mask, lane_mask


# ─── Visualization ──────────────────────────────────────────────────────────

def draw_results(img_bgr, boxes, scores, class_ids, road_mask, lane_mask,
                 obj_list, ratio, dw, dh, seg_alpha=0.4):
    """Draw detection boxes and segmentation overlays on the image."""
    h_orig, w_orig = img_bgr.shape[:2]
    result = img_bgr.copy()

    # Draw segmentation overlay
    if road_mask is not None:
        # Resize seg masks from model output size to letterboxed size
        road_resized = cv2.resize(road_mask, (INPUT_W, INPUT_H),
                                  interpolation=cv2.INTER_NEAREST)
        lane_resized = cv2.resize(lane_mask, (INPUT_W, INPUT_H),
                                  interpolation=cv2.INTER_NEAREST)

        # Remove padding
        top = int(round(dh))
        left = int(round(dw))
        bottom = INPUT_H - int(round(dh))
        right = INPUT_W - int(round(dw))
        road_cropped = road_resized[top:bottom, left:right]
        lane_cropped = lane_resized[top:bottom, left:right]

        # Resize to original image size
        road_orig = cv2.resize(road_cropped, (w_orig, h_orig),
                               interpolation=cv2.INTER_NEAREST)
        lane_orig = cv2.resize(lane_cropped, (w_orig, h_orig),
                               interpolation=cv2.INTER_NEAREST)

        # Apply colored overlays
        overlay = result.copy()
        overlay[road_orig == 1] = SEG_COLORS['road']
        overlay[lane_orig == 1] = SEG_COLORS['lane']
        result = cv2.addWeighted(result, 1 - seg_alpha, overlay, seg_alpha, 0)

    # Draw detection boxes
    for box, score, cls_id in zip(boxes, scores, class_ids):
        cls_name = obj_list[cls_id]
        color = CLASS_COLORS.get(cls_name, (255, 255, 0))

        # Scale box coordinates back to original image
        x1 = int((box[0] - dw) / ratio)
        y1 = int((box[1] - dh) / ratio)
        x2 = int((box[2] - dw) / ratio)
        y2 = int((box[3] - dh) / ratio)

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_orig, x2), min(h_orig, y2)

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name}: {score:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(result, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return result


# ─── Main Inference Pipeline ────────────────────────────────────────────────

class HybridNetsInference:
    """Self-contained ONNX inference engine for HybridNets.

    Usage:
        engine = HybridNetsInference(model_path, config_path)
        detections, road_mask, lane_mask = engine.run(image_bgr)
    """

    def __init__(self, model_path=DEFAULT_MODEL, config_path=DEFAULT_CONFIG,
                 conf_thresh=0.25, iou_thresh=0.45, use_tensorrt=False):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.obj_list = self.config['obj_list']
        self.seg_list = self.config['seg_list']
        self.mean = self.config['mean']
        self.std = self.config['std']
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # Load pre-computed anchors (exact match with ONNX model)
        anchor_path = os.path.join(os.path.dirname(model_path), "anchor_384x640.npy")
        if os.path.exists(anchor_path):
            anchors_raw = np.load(anchor_path)
            if anchors_raw.ndim == 3:
                anchors_raw = anchors_raw[0]  # remove batch dim: (1, N, 4) -> (N, 4)
            # Convert from (x1, y1, x2, y2) to (cx, cy, w, h)
            cx = (anchors_raw[:, 0] + anchors_raw[:, 2]) / 2
            cy = (anchors_raw[:, 1] + anchors_raw[:, 3]) / 2
            w = anchors_raw[:, 2] - anchors_raw[:, 0]
            h = anchors_raw[:, 3] - anchors_raw[:, 1]
            self.anchors = np.stack([cx, cy, w, h], axis=1).astype(np.float32)
            print(f"[HybridNets] Loaded pre-computed anchors: {anchor_path} ({self.anchors.shape})")
        else:
            print(f"[HybridNets] Pre-computed anchors not found, generating...")
            self.anchors = generate_anchors(
                INPUT_H, INPUT_W,
                self.config['anchors_scales'],
                self.config['anchors_ratios']
            )

        # Load ONNX model
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if use_tensorrt:
            providers.insert(0, 'TensorrtExecutionProvider')
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        print(f"[HybridNets] Loaded ONNX model: {model_path}")
        print(f"[HybridNets] Classes: {self.obj_list}")
        print(f"[HybridNets] Seg classes: {self.seg_list}")
        print(f"[HybridNets] Provider: {self.session.get_providers()}")

    def run(self, img_bgr):
        """Run inference on a BGR image.

        Returns:
            detections: list of (box, score, class_name) tuples
            road_mask: binary mask (H, W) for drivable area, at model resolution
            lane_mask: binary mask (H, W) for lane lines, at model resolution
        """
        # Preprocess
        img_input, ratio, dw, dh = preprocess(img_bgr, self.mean, self.std)

        # Run ONNX inference
        t0 = time.time()
        outputs = self.session.run(None, {self.input_name: img_input})
        dt = time.time() - t0

        regression, classification, segmentation = outputs

        # Decode detections
        boxes, scores, class_ids = decode_detections(
            regression, classification, self.anchors,
            self.conf_thresh, self.iou_thresh
        )

        # Decode segmentation
        road_mask, lane_mask = decode_segmentation(
            segmentation, num_classes=len(self.seg_list) + 1
        )

        # Format detections
        detections = []
        for box, score, cls_id in zip(boxes, scores, class_ids):
            cls_name = self.obj_list[cls_id]
            detections.append({
                'box': box,
                'score': float(score),
                'class': cls_name,
                'class_id': int(cls_id)
            })

        return detections, road_mask, lane_mask, ratio, dw, dh, dt

    def run_and_visualize(self, img_bgr):
        """Run inference and return annotated image."""
        detections, road_mask, lane_mask, ratio, dw, dh, dt = self.run(img_bgr)

        boxes = [d['box'] for d in detections]
        scores = [d['score'] for d in detections]
        class_ids = [d['class_id'] for d in detections]

        result = draw_results(img_bgr, boxes, scores, class_ids,
                            road_mask, lane_mask, self.obj_list, ratio, dw, dh)

        # Add FPS overlay
        fps = 1.0 / dt if dt > 0 else 0
        cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return result, detections


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HybridNets ONNX Inference')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='Path to ONNX model')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG,
                        help='Path to config YAML')
    parser.add_argument('--image', type=str, help='Path to a single image')
    parser.add_argument('--image_dir', type=str, help='Path to image directory')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--conf_thresh', type=float, default=0.25,
                        help='Detection confidence threshold')
    parser.add_argument('--iou_thresh', type=float, default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--tensorrt', action='store_true',
                        help='Use TensorRT execution provider')
    args = parser.parse_args()

    # Collect images
    images = []
    if args.image:
        images = [args.image]
    elif args.image_dir:
        images = sorted(glob(os.path.join(args.image_dir, '*.jpg')) +
                        glob(os.path.join(args.image_dir, '*.png')))
    else:
        parser.error("Provide --image or --image_dir")

    os.makedirs(args.output, exist_ok=True)

    # Initialize inference engine
    engine = HybridNetsInference(
        model_path=args.model,
        config_path=args.config,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        use_tensorrt=args.tensorrt
    )

    # Run inference on each image
    for i, img_path in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] {img_path}")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            print(f"  [ERROR] Could not read image: {img_path}")
            continue

        result, detections = engine.run_and_visualize(img)

        # Print detections
        for det in detections:
            print(f"  {det['class']}: {det['score']:.1%}")

        # Save result
        out_path = os.path.join(args.output, os.path.basename(img_path))
        cv2.imwrite(out_path, result)
        print(f"  Saved: {out_path}")

    print(f"\nDone! Results saved to {args.output}/")
