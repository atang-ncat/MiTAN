#!/usr/bin/env python3
"""
HybridNets TensorRT Inference — Native engine loading (no extra dependencies)

Loads the pre-built .engine file directly using TensorRT Python bindings
and ctypes for CUDA memory management (no pycuda/cuda-python needed).
"""

import os
import time
import ctypes
import numpy as np
import cv2
import yaml
import tensorrt as trt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_H = 384
INPUT_W = 640

# ─── CUDA runtime via ctypes ────────────────────────────────────────────────
_cudart = ctypes.CDLL('libcudart.so')

# Properly declare function signatures
_cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cudart.cudaMalloc.restype = ctypes.c_int

_cudart.cudaFree.argtypes = [ctypes.c_void_p]
_cudart.cudaFree.restype = ctypes.c_int

_cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_size_t, ctypes.c_int]
_cudart.cudaMemcpy.restype = ctypes.c_int

_cudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_cudart.cudaStreamCreate.restype = ctypes.c_int

_cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
_cudart.cudaStreamSynchronize.restype = ctypes.c_int

_cudart.cudaDeviceSynchronize.argtypes = []
_cudart.cudaDeviceSynchronize.restype = ctypes.c_int

# CUDA memcpy kinds
MEMCPY_H2D = 1  # cudaMemcpyHostToDevice
MEMCPY_D2H = 2  # cudaMemcpyDeviceToHost


class HybridNetsTRTInference:
    """TensorRT-native inference engine for HybridNets.

    Usage:
        engine = HybridNetsTRTInference(engine_path, config_path)
        detections, road_mask, lane_mask, ratio, dw, dh, dt = engine.run(image_bgr)
    """

    def __init__(self, engine_path, config_path, conf_thresh=0.30, iou_thresh=0.45):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.obj_list = self.config['obj_list']
        self.seg_list = self.config['seg_list']
        self.mean = np.array(self.config['mean'], dtype=np.float32)
        self.std = np.array(self.config['std'], dtype=np.float32)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # Load pre-computed anchors
        anchor_path = os.path.join(os.path.dirname(engine_path), "anchor_384x640.npy")
        if os.path.exists(anchor_path):
            anchors_raw = np.load(anchor_path)
            if anchors_raw.ndim == 3:
                anchors_raw = anchors_raw[0]
            cx = (anchors_raw[:, 0] + anchors_raw[:, 2]) / 2
            cy = (anchors_raw[:, 1] + anchors_raw[:, 3]) / 2
            w = anchors_raw[:, 2] - anchors_raw[:, 0]
            h = anchors_raw[:, 3] - anchors_raw[:, 1]
            self.anchors = np.stack([cx, cy, w, h], axis=1).astype(np.float32)
            print(f"[HybridNets-TRT] Loaded anchors: {anchor_path} ({self.anchors.shape})")
        else:
            raise FileNotFoundError(f"Anchor file not found: {anchor_path}")

        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.trt_engine = runtime.deserialize_cuda_engine(f.read())

        if self.trt_engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        self.context = self.trt_engine.create_execution_context()

        # Create CUDA stream
        self.stream = ctypes.c_void_p()
        ret = _cudart.cudaStreamCreate(ctypes.byref(self.stream))
        if ret != 0:
            print(f"[HybridNets-TRT] WARNING: cudaStreamCreate failed ({ret}), using synchronize")
            self.stream = None

        # Allocate buffers for each tensor
        self.input_name = None
        self.output_info = []

        for i in range(self.trt_engine.num_io_tensors):
            name = self.trt_engine.get_tensor_name(i)
            shape = tuple(self.trt_engine.get_tensor_shape(name))
            dtype = trt.nptype(self.trt_engine.get_tensor_dtype(name))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            mode = self.trt_engine.get_tensor_mode(name)

            # Allocate contiguous host buffer
            host_buf = np.zeros(shape, dtype=dtype)
            host_buf = np.ascontiguousarray(host_buf)

            # Allocate device buffer
            device_ptr = ctypes.c_void_p()
            ret = _cudart.cudaMalloc(ctypes.byref(device_ptr), ctypes.c_size_t(nbytes))
            if ret != 0:
                raise RuntimeError(f"cudaMalloc failed for {name}: error {ret}")

            # Set address for context
            self.context.set_tensor_address(name, device_ptr.value)

            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = shape
                self.input_dtype = dtype
                self.input_host = host_buf
                self.input_device = device_ptr
                self.input_nbytes = nbytes
            else:
                self.output_info.append({
                    'name': name,
                    'shape': shape,
                    'dtype': dtype,
                    'host': host_buf,
                    'device': device_ptr,
                    'nbytes': nbytes,
                })

        print(f"[HybridNets-TRT] Engine loaded: {engine_path}")
        print(f"[HybridNets-TRT] Input: {self.input_name} {self.input_shape} ({self.input_dtype.__name__})")
        for info in self.output_info:
            print(f"[HybridNets-TRT] Output: {info['name']} {info['shape']} ({info['dtype'].__name__})")

    def _preprocess(self, img_bgr):
        """Preprocess image: letterbox + normalize."""
        img_lb, ratio, (dw, dh) = _letterbox(img_bgr, (INPUT_H, INPUT_W))
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        img_float = (img_float - self.mean) / self.std
        img_chw = np.transpose(img_float, (2, 0, 1))
        img_batch = np.expand_dims(img_chw, axis=0).astype(np.float32)
        return np.ascontiguousarray(img_batch), ratio, dw, dh

    def run(self, img_bgr):
        """Run inference on a BGR image.

        Returns same format as HybridNetsInference.run() for compatibility:
            detections, road_mask, lane_mask, ratio, dw, dh, dt
        """
        # Preprocess
        img_input, ratio, dw, dh = self._preprocess(img_bgr)

        # Copy preprocessed input into host buffer, then to device
        np.copyto(self.input_host, img_input)
        _cudart.cudaMemcpy(
            self.input_device,
            ctypes.c_void_p(self.input_host.ctypes.data),
            ctypes.c_size_t(self.input_nbytes),
            ctypes.c_int(MEMCPY_H2D)
        )

        # Run inference
        t0 = time.time()
        if self.stream is not None:
            self.context.execute_async_v3(stream_handle=self.stream.value)
            _cudart.cudaStreamSynchronize(self.stream)
        else:
            # Synchronous fallback: use stream 0
            self.context.execute_async_v3(stream_handle=0)
            _cudart.cudaDeviceSynchronize()
        dt = time.time() - t0

        # Copy outputs from device to host
        outputs = []
        for info in self.output_info:
            _cudart.cudaMemcpy(
                ctypes.c_void_p(info['host'].ctypes.data),
                info['device'],
                ctypes.c_size_t(info['nbytes']),
                ctypes.c_int(MEMCPY_D2H)
            )
            outputs.append(info['host'].copy())

        # Identify outputs by shape: regression, classification, segmentation
        regression = None
        classification = None
        segmentation = None

        for arr in outputs:
            if len(arr.shape) == 4:
                segmentation = arr
            elif len(arr.shape) == 3:
                if arr.shape[-1] == 4:
                    regression = arr
                else:
                    classification = arr

        # Fallback: assume order matches ONNX export
        if regression is None or classification is None or segmentation is None:
            if len(outputs) >= 3:
                regression, classification, segmentation = outputs[0], outputs[1], outputs[2]

        # Decode detections
        boxes, scores, class_ids = _decode_detections(
            regression, classification, self.anchors,
            self.conf_thresh, self.iou_thresh
        )

        # Decode segmentation
        road_mask, lane_mask = _decode_segmentation(
            segmentation, num_classes=len(self.seg_list) + 1
        )

        # Format detections
        detections = []
        for box, score, cls_id in zip(boxes, scores, class_ids):
            detections.append({
                'box': box,
                'score': float(score),
                'class': self.obj_list[cls_id],
                'class_id': int(cls_id)
            })

        return detections, road_mask, lane_mask, ratio, dw, dh, dt

    def __del__(self):
        """Cleanup CUDA memory."""
        try:
            if hasattr(self, 'input_device'):
                _cudart.cudaFree(self.input_device)
            if hasattr(self, 'output_info'):
                for info in self.output_info:
                    _cudart.cudaFree(info['device'])
        except Exception:
            pass


# ─── Helper functions ───────────────────────────────────────────────────────

def _letterbox(img, new_shape=(INPUT_H, INPUT_W), color=(114, 114, 114), stride=128):
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


def _decode_detections(regression, classification, anchors, conf_thresh=0.25,
                       iou_thresh=0.45):
    scores = classification[0]
    reg = regression[0]
    max_scores = np.max(scores, axis=1)
    max_classes = np.argmax(scores, axis=1)
    mask = max_scores > conf_thresh
    if not np.any(mask):
        return [], [], []

    filtered_scores = max_scores[mask]
    filtered_classes = max_classes[mask]
    filtered_reg = reg[mask]
    filtered_anchors = anchors[mask]

    ax, ay, aw, ah = (filtered_anchors[:, 0], filtered_anchors[:, 1],
                      filtered_anchors[:, 2], filtered_anchors[:, 3])
    dx, dy, dw_r, dh_r = (filtered_reg[:, 0], filtered_reg[:, 1],
                           filtered_reg[:, 2], filtered_reg[:, 3])
    cx = ax + dx * aw
    cy = ay + dy * ah
    w = aw * np.exp(dw_r)
    h = ah * np.exp(dh_r)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    final_boxes, final_scores, final_classes = [], [], []
    for cls_id in np.unique(filtered_classes):
        cls_mask = filtered_classes == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = filtered_scores[cls_mask]
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


def _decode_segmentation(seg_output, num_classes=3):
    seg = seg_output[0]
    class_ids = np.argmax(seg, axis=0)
    road_mask = (class_ids == 1).astype(np.uint8)
    lane_mask = (class_ids == 2).astype(np.uint8)
    return road_mask, lane_mask
