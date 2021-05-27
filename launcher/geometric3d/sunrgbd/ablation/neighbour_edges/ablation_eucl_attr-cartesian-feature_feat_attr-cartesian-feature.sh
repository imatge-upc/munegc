#!/bin/bash

script='./../../../../../Fusion2D3DMUNEGC/geometric3d/test.py'
dataset_path='./../../../../../dataset/sunrgbd' 
dataset_folder='h5/h5_3dhha'
test_split='list/test_list.txt'
classname='list/scenes_labels.txt'
pretrain_path='./../../../../../pretrain/geometric_3d/ablation/neighbour_edges/eucl_attr-cartesian-feature_feat_attr-cartesian-feature.pth.tar'
model='multigraphconv_9_16_0,b_0,r_0,pnv_max_0.05_0,multigraphconv_9_16_0,b_0,r_0,pnv_max_0.08_0,multigraphconv_9_32_0,b_0,r_0,pnv_max_0.12_0,multigraphconv_9_64_0,b_0,r_0,pnv_max_0.24_0,multigraphconv_9_128_1,b_1,r_1,gp_avg_1,d_0.2_1,f_19_cp_1'

python $script --batch_size 32 --batch_parts 8 --cuda --nworkers 4 --dataset_path $dataset_path \
		--dataset_folder $dataset_folder --test_split $test_split --classname $classname --nfeatures 3 \
		--range01 --pos_int16 --model_config $model --edge_attr poscart-featureoffsets  --fnet_widths [128] \
		--fnet_llbias  --fnet_tanh --pretrain_path $pretrain_path
