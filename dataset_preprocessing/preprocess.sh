#!/usr/bin/env bash

BASE_DIR="/home/username/dataset/KITTI/"
N_WORKER=6
SPLIT="test"

python mass_production.py \
 --data_path ${BASE_DIR} \
 --n_workers ${N_WORKER} \
 --split ${SPLIT} \
 --process_depth_data \
 --rotation_offset 10 \
 --translation_offset 0.2 \
