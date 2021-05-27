#!/bin/bash

script='./../../../Fusion2D3DMUNEGC/geometric3d/extract_features/extract_features.py'
dataset_path='./../../../dataset/nyuv2' 
dataset_folder='h5/h5_3dhha'
dataset='list/dataset.txt'
test_split='list/test_list.txt'
classname='list/scenes_labels.txt'
pretrain_path='./../../../pretrain/geometric_3d/Nyu_v2_geometric3d_branch.pth.tar'
model='multigraphconv_9_16_0,b_0,r_0,pnv_max_0.05_0,multigraphconv_9_16_0,b_0,r_0,pnv_max_0.08_0,multigraphconv_9_32_0,b_0,r_0,pnv_max_0.12_0,multigraphconv_9_64_0,b_0,r_0,pnv_max_0.24_0,multigraphconv_9_128_1,b_1,r_1,gp_avg_1,d_0.2_1,f_10_cp_1'
h5_feat_3d='h5/h5_feat3d'

python $script --batch_size 32 --batch_parts 8 --cuda --nworkers 4 \
		--dataset_path $dataset_path --dataset_folder $dataset_folder \
		--test_split $test_split --dataset $dataset \
		--classname $classname --nfeatures 3 \
		--range01 --pos_int16 --model_config $model --edge_attr posspherical-featureoffsets  \
		--fnet_widths [128] --fnet_llbias --fnet_tanh --pretrain_path $pretrain_path \
		--check_accuracy --nlayer 17 --h5_feat_3d $h5_feat_3d
