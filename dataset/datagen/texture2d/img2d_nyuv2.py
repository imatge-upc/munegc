"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import numpy as np
import h5py
from tqdm import tqdm
import sys
import warnings

from Fusion2D3DMUNEGC.utilities import utils

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def save_h5_scene_pc(filename, img, label):
    h5 = h5py.File(filename, "w")
    h5.create_dataset(
        "img", data=img,
        dtype='uint8')
    h5.create_dataset(
        "label", data=label,
        dtype='uint8')


def crop(img, new_size):
    img_size = np.shape(img)[0:2]

    start = (np.array(img_size) - np.array(new_size))/2
    end = (img_size-np.floor(start)).astype(int)
    start = np.ceil(start).astype(int)

    if len(np.shape(img)) == 3:
        return img[start[0]:end[0], start[1]:end[1], :]
    else:
        return img[start[0]:end[0], start[1]:end[1]]


if __name__ == "__main__":

    # dataset folder path
    dataset_path = './../../nyuv2'

    # path to the data
    dataset_mat_path = dataset_path + '/nyu_depth_v2_labeled.mat'

    # folder to store the h5 files
    path_h5 = dataset_path + "/h5/h5_2dimg/"
    utils.create_folder(path_h5)

    train_split = utils.read_string_list(dataset_path+"/list/train_list.txt")
    val_split = utils.read_string_list(dataset_path+"/list/test_list.txt")
    scenes_types = utils.read_string_list(dataset_path+"/list/scenes_labels27.txt")
    dataset = train_split + val_split

    print("Loading .mat")
    f = h5py.File(dataset_mat_path, 'r')

    images = np.transpose(np.asarray(f['images']))
    scenes = np.transpose(np.asarray(f['sceneTypes'])).squeeze()
    mapping_10 = np.asarray([10, 4, 1, 9, 10, 8, 10, 10, 10, 5, 10, 10, 10, 7, 10, 10, 2, 10, 3, 6, 10, 10, 10, 10, 10, 10, 10])-1
    newSize = (420, 560)

    for i in tqdm(dataset, ncols=100):
        filename = i
        i = int(i)-1

        img = images[:, :, :, i]

        img = crop(img, newSize)

        label = (''.join(chr(j) for [j] in f[scenes[i]]))
        label_id = mapping_10[scenes_types.index(label)]
        h5_name = filename+".h5"

        save_h5_scene_pc(path_h5 + h5_name, img, label_id)
