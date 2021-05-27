"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import numpy as np
import h5py
from skimage import io
import glob
from tqdm import tqdm

from Fusion2D3DMUNEGC.utilities import utils

import sys
import warnings

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
    dataset_path = '../../sunrgbd'

    # path to the data
    images_list = dataset_path + '/list/sun_list.txt'
    label_list = dataset_path + '/list/scenes_labels.txt'

    # folder to store the h5 files
    path_h5 = dataset_path + "/h5/h5_2dimg/"

    utils.create_folder(path_h5)

    images = utils.read_string_list(images_list)

    dataset_labels = utils.read_string_list(label_list)
    newSize = (420, 560)

    for i in tqdm(range(0, len(images)), ncols=100):
        img_folder = dataset_path+"/"+images[i]

        rgb_path = glob.glob(img_folder+'/image/*.jpg')[0]

        img = io.imread(rgb_path)

        readlabel = open(img_folder + "/scene.txt", "r")
        label = readlabel.read()

        img = crop(img, newSize)

        # Save h5_file
        h5_name = (images[i].replace("/", "_")+".h5").replace("_.h5", ".h5")

        label_id = dataset_labels.index(label)

        save_h5_scene_pc(path_h5 + h5_name, img, label_id)
