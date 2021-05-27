"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import numpy as np
import h5py
from skimage import io
import glob
from tqdm import tqdm
import torch

from Fusion2D3DMUNEGC.utilities import utils

def read_depth_sunrgbd(depth_path):
    depthVis = io.imread(depth_path)
    depthInPaint = ((depthVis >> 3) | (depthVis << 13))/1000
    depthInPaint[np.where(depthInPaint > 8)] = 8
    return depthInPaint


def create_point_cloud_depth(depth, fx, fy, cx, cy):
    depth_shape = depth.shape
    [x_d, y_d] = np.meshgrid(range(0, depth_shape[1]), range(0, depth_shape[0]))
    x3 = np.divide(np.multiply((x_d-cx), depth), fx)
    y3 = np.divide(np.multiply((y_d-cy), depth), fy)
    z3 = depth

    return np.stack((x3, y3, z3), axis=2)


def matrix3d2vector(matrix):
    return np.reshape(matrix, (-1, 3))


def save_h5_scene_pc(filename, img, label):
    h5 = h5py.File(filename, "w")
    h5.create_dataset(
        "points", data=img,
        dtype='float')
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
    path_h5 = dataset_path + '/h5/h5_feat2d/'
    utils.create_folder(path_h5)

    images = utils.read_string_list(images_list)

    dataset_labels = utils.read_string_list(label_list)
    newSize = (420, 560)

    for i in tqdm(range(0, len(images)), ncols=100):
        img_folder = dataset_path+"/" + images[i]

        depth_path = glob.glob(img_folder + "/depth_bfx/*.png")[0]

        depth_img = read_depth_sunrgbd(depth_path)
        intrinsic = np.loadtxt(img_folder + '/intrinsics.txt')

        readlabel = open(img_folder + "/scene.txt", "r")
        label = readlabel.read()

        points = create_point_cloud_depth(depth_img, intrinsic.item(0),
                                          intrinsic.item(4), intrinsic.item(2), intrinsic.item(5))

        points = crop(points, newSize)

        points_average = torch.nn.AvgPool2d(32, count_include_pad=False, ceil_mode=True)
        points_torch = torch.tensor(points).permute(2, 0, 1)

        points_torch_32 = points_average(points_torch)
        points_d32 = points_torch_32.data.numpy()
        points_d32 = np.round(points_d32, 3)

        h5_name = (images[i].replace("/", "_")+".h5").replace("_.h5", ".h5")

        label_id = dataset_labels.index(label)

        points_d32_vector = matrix3d2vector(points_d32)

        save_h5_scene_pc(path_h5 + h5_name, points_d32_vector, label_id)
