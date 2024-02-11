#!/bin/bash

# Define the location and filename for the weights
WEIGHTS_PATH="/code/catkin_ws/src/dt-core/packages/yolov3.weights" # Adjust this path as needed

# URL to your YOLO weights
WEIGHTS_URL="https://drive.google.com/uc?export=download&id=1Eg_kl6C9ZHaLc-SkvF77WFoWYwVZVaFX"

# Check if the weights file already exists to avoid re-downloading it
if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "Downloading YOLO weights..."
    wget --no-check-certificate "$WEIGHTS_URL" -O "$WEIGHTS_PATH"
else
    echo "YOLO weights already present, skipping download."
fi