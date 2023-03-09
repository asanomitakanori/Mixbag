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
    --name=my_docker \
    --env-file=.devcontainer/.env \
    --volume=$PWD:/workspace/online-pseudo-labeling/ \
    --mount=source=/raid/matsuo/dataset/,target=/workspace/dataset/,type=bind,consistency=cached \
    my_docker:latest \
    fish
