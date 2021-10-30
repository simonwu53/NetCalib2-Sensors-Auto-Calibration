#!/usr/bin/env bash

KITTI_DIR="/home/username/dataset/KITTI/"

source /home/username/Projects/venvs/pytorch/bin/activate

python train.py \
--dataset ${KITTI_DIR} \
--batch 2 \
--model 1 \
--epoch 20 \
--lr 1e-5 \
--patience 3 \
--lr_factor 0.5 \
--loss_a 1.25 \
--loss_b 1.75 \
--loss_c 1.0 \

# --exp_name HOPE-10-0.2 \
# --ckpt /home/username/OneDriveUT/projects/autocalibration_project/results/HOPE-10-0.2/ckpt/Epoch65_val_0.2983.tar \
# --ckpt_no_lr
