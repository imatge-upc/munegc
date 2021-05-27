#!/bin/bash

script='./../../../Fusion2D3DMUNEGC/geometric3d/train.py'
seed=464
odir='./../../../results'
dataset='sun'
dataset_path='./../../../dataset/sunrgbd' 
dataset_folder='h5/h5_3dhha'
train_split='list/train_list.txt'
val_split='list/test_list.txt'
classname='list/scenes_labels.txt'
weights='list/weights/weights_inverse_frequency_norm.txt'
model='multigraphconv_9_16_0,b_0,r_0,pnv_max_0.05_0,multigraphconv_9_16_0,b_0,r_0,pnv_max_0.08_0,multigraphconv_9_32_0,b_0,r_0,pnv_max_0.12_0,multigraphconv_9_64_0,b_0,r_0,pnv_max_0.24_0,multigraphconv_9_128_1,b_1,r_1,gp_avg_1,d_0.2_1,f_19_cp_1'
exp_name='Sunrgbd_geometric3d_branch'

python $script --exp_name $exp_name --optim radam --lr 0.001 --wd 0.0001 \
		--betas '(0.9, 0.999)' --epochs 200 --batch_size 32 --batch_parts 8 \
		--cuda --nworkers 4 --dataset_path $dataset_path --dataset_folder $dataset_folder \
		--train_split $train_split --val_split $val_split --classname $classname --nfeatures 3 \
		--dataset $dataset --weights $weights --range01 --pos_int16 --random_crop --factor_rand \
		--factor 0.875 --odir $odir --seed $seed  --model_config $model --edge_attr posspherical-featureoffsets \
		--fnet_widths [128] --fnet_llbias  --fnet_tanh  --pc_augm_input_dropout 0.2 --pc_augm_rot \
		--pc_augm_mirror_prob 0.5
