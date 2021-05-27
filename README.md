MUNEGC: 2D-3D Geometric Fusion Network using Multi-Neighbourhood Graph Convolution for RGB-D Indoor Scene Classification
=========

See our project website [here](https://imatge-upc.github.io/munegc/).

## Code Structure


* `./dataset/` - Scripts to generate the dataset and the dataset itself.
* `./Fusion2D3DMUNEGC` - Implementation of the network and the proposed operations. 
* `./launcher` - Scripts to train and test the proposed network.
* `./results` - Stores the results obtained during training.
* `./pretrain/*` - Pretrained weights for each branch.

## Requirements

1. Install **Pytorch 1.8.1** following the oficial [documentation](https://pytorch.org).
2. Install **Pytorch-Geometric 1.7.0, torch-cluster 1.5.9, torch-scatter 2.0.6 and torch-sparse 0.6.9** following the oficial [documentation](https://github.com/rusty1s/pytorch_geometric).
3. Install the MUNEGC package running the following line `pip install -e .` at the root folder of the project.

This code has been tested in Ubuntu-18.04 using Python3.6, Cuda11.1 and the previous mentioned versions of Pytorch and Pytorch Geometric.

## Datasets

You can download our pre-processed dataset [here](https://drive.google.com/drive/folders/1mor4CyWsagyTFq_qaunqv-Iv4SHhkUBq?usp=sharing). If you want to create your own version of the dataset, you can download the official NYUV2 dataset [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and the official SUNRGBD dataset [here](http://rgbd.cs.princeton.edu/). We also provide the scripts to generate the data for each of the branches in `dataset/datagen`. These scripts expect to find the raw data in `dataset/nyuv2` or `dataset/sunrgbd`. To generate the HHA features, we have used the implementation done by charlesCXK, included in this repository. You can find the original implementation [here](https://github.com/charlesCXK/Depth2HHA-python). 


## Test

To reproduce the paper's results you can download the pre-trained models [here](https://drive.google.com/drive/folders/1mor4CyWsagyTFq_qaunqv-Iv4SHhkUBq?usp=sharing). You can find the scripts needed for run the tests for each of the branches in the folder `launcher`.   


## Training

As explained in the paper, the branches of the network are trained independently. To train the **2D Texture and 3D Geometric** branches, you need to run the corresponding training script located in the `launcher` folder. To train the **Fusion Stage**, first we have to extract the features obtained by both branches. 

1. In order to obtain the **2D texture features**, first we have to run the scripts located in `dataset/datagen/fusion2d3d`. These scripts are going to generate the 3D points for each of the features obtained by the **2D Texture branch**. To add the features to the obtained 3D points, you have to run the `extract_features.sh` script located in the `launcher/texture2d` folder.

2. In order to obtain the **3D Geometric features** you have to run the `extract_features.sh` script located in the `launcher/geometric3d` folder.


NOTE: Please check `graph_model.py` in order to understand how you can define your own architecture using the provided operations.


## Issues

Due to the existence of some non-deterministic operations in pytorch, as explained [here](https://pytorch.org/docs/stable/notes/randomness.html), some results may not be reproducible or give slightly different values. This effect also appears when you use different model of GPUs to train and test the network.

## Citation

```
@article{MOSELLAMONTORO2021,
title = {2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification},
journal = {Information Fusion},
volume = {76},
pages = {46-54},
year = {2021},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2021.05.002},
author = {Albert Mosella-Montoro and Javier Ruiz-Hidalgo},
}
```

## Contact

For questions and suggestions send an e-mail to albert.mosella@upc.edu.
