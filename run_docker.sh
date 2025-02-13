#!/usr/bin/env bash

# Get the directory of this script
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

docker run -it --rm --gpus all \
    --name planning_project \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY=$DISPLAY \
    -v $PROJECT_DIR:/workspace \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /dev/dri:/dev/dri \
    -w /workspace \
    --network host \
    bubble_cdf_planner:latest \