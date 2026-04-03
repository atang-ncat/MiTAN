#!/bin/bash
# Convert HybridNets ONNX model to TensorRT engine (FP16)
# Run this INSIDE the Isaac ROS Docker container.
#
# Usage:  bash convert_hybridnets_trt.sh
#
# The engine file is hardware-specific (built for your exact Orin GPU).
# This only needs to be done once.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ONNX_PATH="${SCRIPT_DIR}/weights/hybridnets_384x640.onnx"
ENGINE_PATH="${SCRIPT_DIR}/weights/hybridnets_384x640.engine"

if [ ! -f "$ONNX_PATH" ]; then
    echo "ERROR: ONNX model not found at $ONNX_PATH"
    exit 1
fi

if [ -f "$ENGINE_PATH" ]; then
    echo "WARNING: Engine already exists at $ENGINE_PATH"
    read -p "Overwrite? [y/N] " yn
    case $yn in
        [Yy]* ) ;;
        * ) echo "Aborted."; exit 0;;
    esac
fi

echo "Converting ONNX → TensorRT FP16 engine..."
echo "  Input:  $ONNX_PATH"
echo "  Output: $ENGINE_PATH"
echo ""
echo "This may take 2-5 minutes on the Orin..."
echo ""

/usr/src/tensorrt/bin/trtexec \
    --onnx="$ONNX_PATH" \
    --saveEngine="$ENGINE_PATH" \
    --fp16 \
    --memPoolSize=workspace:16384MiB \
    --verbose

echo ""
echo "Done! Engine saved to: $ENGINE_PATH"
echo "Size: $(du -h "$ENGINE_PATH" | cut -f1)"
