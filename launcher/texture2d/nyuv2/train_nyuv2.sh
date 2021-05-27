#!/bin/bash

script='./../../../Fusion2D3DMUNEGC/texture2d/train.py'
dataset_path='./../../../dataset/nyuv2' 
dataset_folder='h5/h5_2dimg'
train_split='list/train_list.txt'
val_split='list/test_list.txt'
classname='list/scenes_labels.txt'
weights='list/weights/weights_inverse_frequency_norm.txt'
pretrain_path='./../../../pretrain/texture_2d/Sunrgbd_texture2d_branch.pth.tar'
odir='./../../../results/'
exp_name='Nyu_v2_texture2d_branch'

python $script --optim 'sgd' --lr 0.001 --wd 0.0001 --momentum 0.9 --epochs 100 \
		--batch_size 16 --cuda --nworkers 4 --range01 --dataset 'nyu_v2' \
		--dataset_path $dataset_path --dataset_folder $dataset_folder \
		--train_split $train_split --val_split $val_split --classname $classname \
		--nclass 10 --weights $weights --transfer_learning --tl_nclass 19 \
		--tl_path $pretrain_path --odir $odir --exp_name $exp_name --seed 131
