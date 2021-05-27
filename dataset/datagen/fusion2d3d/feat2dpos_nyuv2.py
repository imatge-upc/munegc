"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import numpy as np
import h5py
from tqdm import tqdm
import torch

from Fusion2D3DMUNEGC.utilities import utils

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
    dataset_path = '../../nyuv2'

    # path to the data
    dataset_mat_path = dataset_path + '/nyu_depth_v2_labeled.mat'

    # folder to store the h5 files
    path_h5 = dataset_path + '/h5/h5_feat2d/'
    utils.create_folder(path_h5)

    train_split = utils.read_string_list(dataset_path+"/list/train_list.txt")
    val_split = utils.read_string_list(dataset_path+"/list/test_list.txt")
    scenes_types = utils.read_string_list(dataset_path+"/list/scenes_labels27.txt")
    dataset = train_split + val_split

    print("Loading .mat")
    f = h5py.File(dataset_mat_path, 'r')

    depths = np.round(np.transpose(np.asarray(f['depths'])), 4)
    scenes = np.transpose(np.asarray(f['sceneTypes'])).squeeze()

    mapping_10 = np.asarray([10, 4, 1, 9, 10, 8, 10, 10, 10, 5, 10, 10, 10, 7,
                             10, 10, 2, 10, 3, 6, 10, 10, 10, 10, 10, 10, 10])-1


    # intrinsics
    fx = 5.1885790117450188e+02
    fy = 5.1946961112127485e+02
    cx = 3.2558244941119034e+02
    cy = 2.5373616633400465e+02

    newSize = (420, 560)

    for i in tqdm(dataset, ncols=100):
        filename = i
        i = int(i)-1
        depth_img = depths[:, :, i]
        points = create_point_cloud_depth(depth_img, fx, fy, cx, cy)

        points = crop(points, newSize)

        points_average = torch.nn.AvgPool2d(32, count_include_pad=False, ceil_mode=True)
        points_torch = torch.tensor(points).permute(2, 0, 1)

        points_torch_32 = points_average(points_torch)
        points_d32 = points_torch_32.data.numpy()
        points_d32 = np.round(points_d32, 3)

        h5_name = filename+'.h5'

        label = (''.join(chr(j) for [j] in f[scenes[i]]))
        label_id = mapping_10[scenes_types.index(label)]

        points_d32_vector = matrix3d2vector(points_d32)

        save_h5_scene_pc(path_h5 + h5_name, points_d32_vector, label_id)
