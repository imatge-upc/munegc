#!/bin/bash

script='./../../../Fusion2D3DMUNEGC/fusion2d3d/train.py'
seed=716
odir='./../../../results'
dataset='sun'
dataset_path='./../../../dataset/sunrgbd' 
dataset_folder_b1='h5/h5_feat3d'
dataset_folder_b2='h5/h5_feat2d'
train_split='list/train_list.txt'
val_split='list/test_list.txt'
classname='list/scenes_labels.txt'
weights='list/weights/weights_inverse_frequency_norm.txt'
classification_model='b,r,gp_avg,d_0.5,f_19_cp_1'
exp_name='Sunrgbd_2D-3D-fusion_branch'

python $script --exp_name $exp_name \
		--optim radam --lr 0.001 --wd 0.0001 \
		--betas '(0.9, 0.999)' --epochs 20 \
		--batch_size 32 --cuda --nworkers 4 \
		--pos_int16 --dataset_path $dataset_path \
		--dataset_folder_b1 $dataset_folder_b1 \
		--dataset_folder_b2 $dataset_folder_b2 \
		--nfeatures_b1 128 --nfeatures_b2 512 \
		--proj_b1 --proj_b2 --features_proj_b1 256 \
		--features_proj_b1 256 --rad_fuse_pool 0.24 \
		--train_split $train_split --val_split $val_split \
		--classname $classname --dataset $dataset \
		--weights $weights --odir $odir --seed $seed \
		--classification_model $classification_model
