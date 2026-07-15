#!/bin/bash

set -e

PROJECT_DIR="$HOME/Documents/PhD_Dualarm/dualarm_isaac_gym"
IMAGE_NAME="isaacgym"
CONTAINER_NAME="dualarm_isaac_gym_container"

xhost +local:docker || true

# Check if the container is already running
if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    echo "Container is already running. Attaching a new terminal..."
    docker exec -it -e DISPLAY=$DISPLAY ${CONTAINER_NAME} /bin/bash
else
    echo "Starting a new container session..."
    docker run -it --rm \
      --gpus all \
      --network host \
      --name ${CONTAINER_NAME} \
      -e DISPLAY=$DISPLAY \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      -v ${PROJECT_DIR}:/workspace/dualarm_isaac_gym \
      ${IMAGE_NAME} \
      /bin/bash
fi