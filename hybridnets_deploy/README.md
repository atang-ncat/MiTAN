# HybridNets Deployment Package

Self-contained package for running HybridNets inference on Jetson AGX Orin.
**No PyTorch or HybridNets source code required.**

## Contents

```
hybridnets_deploy/
├── weights/
│   ├── hybridnets_384x640.onnx     # ONNX model (2.2 MB)
│   └── hybridnets_best.pth         # PyTorch checkpoint (fallback, 151 MB)
├── config/
│   └── our_dataset.yml             # Classes, anchors, normalization params
├── hybridnets_inference.py         # Standalone inference script + importable class
└── README.md                       # This file
```

## Quick Start

### 1. Install Dependencies on Jetson

```bash
pip install onnxruntime-gpu opencv-python numpy pyyaml
# For TensorRT acceleration (recommended on Jetson):
# onnxruntime-gpu on Jetson comes with TensorRT support
```

### 2. Run on a Single Image

```bash
python3 hybridnets_inference.py --image /path/to/frame.jpg --output results/
```

### 3. Run on a Folder of Images

```bash
python3 hybridnets_inference.py --image_dir /path/to/frames/ --output results/
```

### 4. Use TensorRT Acceleration

```bash
python3 hybridnets_inference.py --image /path/to/frame.jpg --tensorrt
```

## Integration into ROS 1 Node

Import the `HybridNetsInference` class into your node:

```python
#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from hybridnets_inference import HybridNetsInference

class HybridNetsNode:
    def __init__(self):
        self.engine = HybridNetsInference(
            model_path='/path/to/weights/hybridnets_384x640.onnx',
            config_path='/path/to/config/our_dataset.yml',
            conf_thresh=0.3
        )
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber('/camera/color/image_raw', Image, self.callback)
        self.pub_vis = rospy.Publisher('/hybridnets/visualization', Image, queue_size=1)

    def callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        result, detections = self.engine.run_and_visualize(img)
        self.pub_vis.publish(self.bridge.cv2_to_imgmsg(result, 'bgr8'))

if __name__ == '__main__':
    rospy.init_node('hybridnets_node')
    node = HybridNetsNode()
    rospy.spin()
```

## Model Details

| Metric | Value |
|--------|-------|
| Input size | 384 × 640 (H × W) |
| Detection classes | person, robot, traffic_light_green, traffic_light_off, traffic_light_red |
| Segmentation classes | road, lane |
| mAP@0.5 | 73.3% |
| Road IoU | 94.1% |
| Lane IoU | 76.9% |
| ONNX opset | 18 |

## Confidence Threshold Tuning

- Default: `0.25` (shows more detections)
- Recommended for deployment: `0.30 - 0.40` (fewer false positives)
- Fine-tune using `--conf_thresh` flag
