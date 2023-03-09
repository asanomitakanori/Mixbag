#!/bin/bash
docker build \
    --pull \
    --rm \
    -f "Dockerfile" \
    --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg USER=asanomi --build-arg PASSWORD=1010 \
    -t \
    asanomi_docker3:latest "."