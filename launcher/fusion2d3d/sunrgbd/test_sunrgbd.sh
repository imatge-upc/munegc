#!/bin/bash

script='./../../../Fusion2D3DMUNEGC/fusion2d3d/test.py'
dataset_path='./../../../dataset/sunrgbd' 
dataset_folder_b1='h5/h5_feat3d'
dataset_folder_b2='h5/h5_feat2d'
test_split='list/test_list.txt'
classname='list/scenes_labels.txt'
pretrain_path='./../../../pretrain/fusion2d3dmunegc/Sunrgbd_fusion2d3d_branch.pth.tar'
classification_model='b,r,gp_avg,d_0.5,f_19_cp_1'


python -W ignore  $script --batch_size 32 --cuda --nworkers 4 \
		--dataset_path $dataset_path  --dataset_folder_b1 $dataset_folder_b1 \
		--dataset_folder_b2 $dataset_folder_b2 --proj_b1 --proj_b2 \
		--features_proj_b1 256 --features_proj_b1 256 --rad_fuse_pool 0.24 \
		--test_split $test_split  --classname $classname \
		--classification_model $classification_model --pretrain_path $pretrain_path

