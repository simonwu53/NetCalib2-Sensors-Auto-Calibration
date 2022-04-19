#!/usr/bin/env bash

BASE_DIR="/home/username/dataset/KITTI_root/"
N_WORKER=6
SPLIT="train"

python preprocess.py \
 --data_path ${BASE_DIR} \
 --n_workers ${N_WORKER} \
 --split ${SPLIT} \
 --rotation_offset 10 \
 --translation_offset 0.2 \
 --no_cam_depth # use this option to generate LiDAR depth only (no camera depth!)
