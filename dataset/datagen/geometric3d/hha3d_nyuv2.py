"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import numpy as np
from skimage.transform import rescale
import h5py
from tqdm import tqdm
import sys
import warnings
from Fusion2D3DMUNEGC.utilities import utils
sys.path.append("../depth2hha/")
from utils.rgbd_util import *
from utils.getCameraParam import *
from getHHA import *

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def create_point_cloud_depth(depth, fx, fy, cx, cy):
    depth_shape = depth.shape
    [x_d, y_d] = np.meshgrid(range(0, depth_shape[1]), range(0, depth_shape[0]))
    x3 = np.divide(np.multiply((x_d-cx), depth), fx)
    y3 = np.divide(np.multiply((y_d-cy), depth), fy)
    z3 = depth

    return np.stack((x3, y3, z3), axis=2)


def save_h5_scene_pc(filename, points, feat, label):
    h5 = h5py.File(filename, "w")
    h5.create_dataset(
        "points", data=points,
        compression='gzip', compression_opts=4,
        dtype='int16')
    h5.create_dataset(
        "feat", data=feat,
        compression='gzip', compression_opts=4,
        dtype='uint8')
    h5.create_dataset(
        "label", data=label,
        dtype='uint8')

    h5.flush()
    h5.close()


def crop(img, new_size):
    img_size = np.shape(img)[0:2]

    start = (np.array(img_size) - np.array(new_size))/2
    end = (img_size-np.floor(start)).astype(int)
    start = np.ceil(start).astype(int)

    if len(np.shape(img)) == 3:
        return img[start[0]:end[0], start[1]:end[1], :]
    else:
        return img[start[0]:end[0], start[1]:end[1]]


def generate_hha(i, ROOT_PATH, depths, scenes, mapping_10, scenes_types, hha_cam_matrix, path_h5):
    filename = i
    i = int(i)-1

    newSize = (420, 560)

    depth_img = depths[:, :, i]

    points = create_point_cloud_depth(depth_img, hha_cam_matrix.item(
        0), hha_cam_matrix.item(4), hha_cam_matrix.item(2), hha_cam_matrix.item(5))

    hha = getHHA(hha_cam_matrix, depth_img, depth_img)
    hha = hha[..., ::-1]  # to rgb instead of bgr

    hha = crop(hha, newSize)
    points = crop(points, newSize)

    points_d8 = rescale(points, 1/8, anti_aliasing=False, multichannel=True, anti_aliasing_sigma=False)

    hha_d8 = rescale(hha, 1/8, anti_aliasing=False, multichannel=True, anti_aliasing_sigma=False)

    points_d8 = (np.round(points_d8, 3) * 1000).astype(np.int16)

    hha_d8 = np.clip(np.rint(hha_d8 * 255), 0, 255).astype(np.uint8)

    h5_name = str(filename) + '.h5'

    label = (''.join(chr(j) for [j] in f[scenes[i]]))
    label_id = mapping_10[scenes_types.index(label)]

    save_h5_scene_pc(path_h5 + h5_name, points_d8.astype(np.int16), hha_d8.astype(np.uint8), label_id)


if __name__ == "__main__":

    # dataset folder path
    dataset_path = '../../nyuv2'
    # path to the data
    dataset_mat_path = dataset_path + '/nyu_depth_v2_labeled.mat'

    # folder to store the h5 files
    path_h5 = dataset_path + '/h5/h5_3dhha/'

    utils.create_folder(path_h5)

    train_split = utils.read_string_list(dataset_path + "/list/train_list.txt")
    val_split = utils.read_string_list(dataset_path + "/list/test_list.txt")
    scenes_types = utils.read_string_list(dataset_path + "/list/scenes_labels27.txt")
    dataset = train_split + val_split

    print("Loading .mat")
    f = h5py.File(dataset_mat_path, 'r')

    depths = np.round(np.transpose(np.asarray(f['depths'])), 4)
    scenes = np.transpose(np.asarray(f['sceneTypes'])).squeeze()

    # map to 10 class config
    mapping_10 = np.asarray([10, 4, 1, 9, 10, 8, 10, 10, 10, 5, 10, 10, 10, 7,
                             10, 10, 2, 10, 3, 6, 10, 10, 10, 10, 10, 10, 10])-1

    # intrinsics
    fx = 5.1885790117450188e+02
    fy = 5.1946961112127485e+02
    cx = 3.2558244941119034e+02
    cy = 2.5373616633400465e+02
    hha_cam_matrix = get_cam_matrix_hha(fx, fy, cx, cy)

    for i in tqdm(dataset):
        generate_hha(i, dataset_path, depths, scenes, mapping_10, scenes_types, hha_cam_matrix, path_h5)
