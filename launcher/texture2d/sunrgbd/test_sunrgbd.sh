#!/bin/bash

script='./../../../Fusion2D3DMUNEGC/texture2d/test.py'
dataset_path='./../../../dataset/sunrgbd' 
dataset_folder='h5/h5_2dimg'
test_split='list/test_list.txt'
classname='list/scenes_labels.txt'
pretrain_path='./../../../pretrain/texture_2d/Sunrgbd_texture2d_branch.pth.tar'

python $script --batch_size 16 --cuda --nworkers 4 --range01 --dataset_path $dataset_path \
		--dataset_folder $dataset_folder --test_split $test_split \
		--classname $classname --nclass 19 --pretrain_path $pretrain_path  
