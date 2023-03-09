#!/bin/bash
# docker exec -itd my_docker jupyter-lab --no-browser --port=${@-8888} --ip=0.0.0.0 --allow-root --NotebookApp.token=''
docker exec -itd my_docker jupyter-lab --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token=''
