#!/bin/bash
docker run \
    -d \
    --init \
    --rm \
    -p 15000:5000 \
    -p 16006:6006 \
    -p 18501-18511:8501-8511 \
    -p 1888:8888 \
    -p 8888:8888 \
    -it \
    --gpus=all \
    --ipc=host \
    --name=asanomi_docker3 \
    --env-file=.devcontainer/.env \
    --volume=$PWD:/workspace/ \
    asanomi_docker3:latest \
    fish
