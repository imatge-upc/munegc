"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import numpy as np
from skimage.transform import rescale
import h5py
from skimage import io
import glob
from tqdm import tqdm
import sys
import warnings
from multiprocessing import Pool

from Fusion2D3DMUNEGC.utilities import utils

sys.path.append("../depth2hha/")
from getHHA import *
from utils.getCameraParam import *
from utils.rgbd_util import *


if not sys.warnoptions:
    warnings.simplefilter("ignore")


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


def update(*a):
    pbar.update()


def generate_hha(i, ROOT_PATH, images, dataset_labels, path_h5):

    newSize = (420, 560)
    img_folder = ROOT_PATH+"/"+images[i]

    depth_path = glob.glob(img_folder+"/depth_bfx/*.png")[0]

    depth_img = read_depth_sunrgbd(depth_path)

    intrinsic = np.loadtxt(img_folder + '/intrinsics.txt')

    readlabel = open(img_folder + "/scene.txt", "r")
    label = readlabel.read()

    points = create_point_cloud_depth(depth_img, intrinsic.item(
        0), intrinsic.item(4), intrinsic.item(2), intrinsic.item(5))

    cam_matrix = np.array([[intrinsic.item(0), 0, intrinsic.item(4)], [
                          0, intrinsic.item(2), intrinsic.item(5)], [0, 0, 1]])
    hha = getHHA(cam_matrix, depth_img, depth_img)
    hha = hha[..., ::-1] # to rgb instead of bgr

    points_d8 = rescale(points, 1/8, anti_aliasing=False, multichannel=True, anti_aliasing_sigma=False)

    hha_d8 = rescale(hha, 1/8, anti_aliasing=False, multichannel=True, anti_aliasing_sigma=False)

    points_d8 = (np.round(points_d8, 3) * 1000).astype(np.int16)

    hha_d8 = np.clip(np.rint(hha_d8*255), 0 ,255).astype(np.uint8)

    # Save h5_file
    h5_name = (images[i].replace("/", "_")+".h5").replace("_.h5", ".h5")

    label_id = dataset_labels.index(label)

    save_h5_scene_pc(path_h5 + h5_name, points_d8.astype(np.int16), hha_d8.astype(np.uint8), label_id)

    return True


if __name__ == "__main__":

    # dataset folder path
    dataset_path = '../../sunrgbd'

    # path to the data
    images_list = dataset_path + '/list/sun_list.txt'
    label_list = dataset_path + '/list/scenes_labels.txt'

    # folder to store the h5 files
    path_h5 = dataset_path + '/h5/h5_3dhha/'

    utils.create_folder(path_h5)

    images = utils.read_string_list(images_list)

    dataset_labels = utils.read_string_list(label_list)

    pbar = tqdm(total=int(len(images)))
    processNum = 32
    pool = Pool(processNum)

    start = 0
    end = len(images)

    for i in range(start, end):
        pool.apply_async(generate_hha, args=(i, dataset_path, images, dataset_labels, path_h5), callback=update)
    pool.close()
    pool.join()
