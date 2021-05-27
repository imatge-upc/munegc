#!/bin/bash

script='./../../../Fusion2D3DMUNEGC/texture2d/extract_features/extract_features.py'
dataset_path='./../../../dataset/sunrgbd' 
dataset_folder='h5/h5_2dimg'
dataset='list/dataset.txt'
test_split='list/test_list.txt'
classname='list/scenes_labels.txt'
nclass=19
pretrain_path='./../../../pretrain/texture_2d/Sunrgbd_texture2d_branch.pth.tar'
h5_feat_2d='h5/h5_feat2d'

python $script --batch_size 16 --cuda --nworkers 0 \
		--range01 --dataset_path $dataset_path \
		--dataset_folder $dataset_folder --dataset $dataset \
		--test_split $test_split  --classname $classname  \
		--nclass $nclass --pretrain_path $pretrain_path \
		--check_accuracy --nlayer 7 \
		--h5_feat_2d $h5_feat_2d  
