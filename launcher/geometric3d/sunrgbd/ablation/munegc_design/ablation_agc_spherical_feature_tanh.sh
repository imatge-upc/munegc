#!/bin/bash

script='./../../../../../Fusion2D3DMUNEGC/geometric3d/test.py'
dataset_path='./../../../../../dataset/sunrgbd' 
dataset_folder='h5/h5_3dhha'
test_split='list/test_list.txt'
classname='list/scenes_labels.txt'
pretrain_path='./../../../../../pretrain/geometric_3d/ablation/munegc_design/agc_spherical_feature_tanh.pth.tar'
model='ggknn_9_0,agc_16_0,b_0,r_0,pnvknn_max_0.05_9_0,agc_16_0,b_0,r_0,pnvknn_max_0.08_9_0,agc_32_0,b_0,r_0,pnvknn_max_0.12_9_0,agc_64_0,b_0,r_0,pnvknn_max_0.24_9_0,agc_128_1,b_1,r_1,gp_avg_1,d_0.2_1,f_19_cp_1'

python $script --batch_size 32 --batch_parts 8 --cuda --nworkers 4 --dataset_path $dataset_path \
		--dataset_folder $dataset_folder --test_split $test_split --classname $classname --nfeatures 3 \
		--range01 --pos_int16 --model_config $model --edge_attr posspherical-featureoffsets --fnet_widths [128] \
		--fnet_llbias  --fnet_tanh --pretrain_path $pretrain_path
