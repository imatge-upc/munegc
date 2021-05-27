"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import torch
from torch_geometric.data import Data
import math
import random
import numpy as np
import transforms3d
import os
import h5py
import torch_cluster
from Fusion2D3DMUNEGC.utilities import utils


def dropout(P, F, p):
    idx = random.sample(range(P.shape[0]), int(math.ceil((1-p)*P.shape[0])))
    return P[idx, :], F[idx, :] if F is not None else None


def random_crop_3D(P, F, factor):
    npoints = P.shape[0]
    n_points_after_crop = np.round(npoints*factor).astype(np.int)

    points_max = (P.max(axis=0)*1000).astype(np.int)
    points_min = (P.min(axis=0)*1000).astype(np.int)

    centroid = np.asarray([np.random.randint(low=points_min[0], high=points_max[0], dtype=int),
                           np.random.randint(low=points_min[1], high=points_max[1], dtype=int),
                           np.random.randint(low=points_min[2], high=points_max[2], dtype=int)])

    centroid = centroid.astype(np.float32)/1000

    rad = 0.1
    inc = 0.2

    npoints_inside_sphere = 0

    x = torch.from_numpy(P)
    y = torch.from_numpy(centroid).unsqueeze(0)
    while npoints_inside_sphere < n_points_after_crop:
        _, crop = torch_cluster.radius(x, y, rad, max_num_neighbors=n_points_after_crop)

        npoints_inside_sphere = len(crop)

        rad = np.round(rad + inc, 1)

    return P[crop.numpy()], F[crop.numpy()]

#
# Modified version of PCH5Dataset that returns the indx of the file of each sample.
#

class PCH5Dataset(torch.utils.data.Dataset):
    def __init__(self, root_path, h5_folder, split,
                 transform3d=None,
                 range01=False, pos_int16=False,
                 random_crop=False, factor_rand=False, factor=1):

        self.root_path = root_path

        self.h5_path = os.path.join(self.root_path, h5_folder)
        self.split = utils.read_string_list(os.path.join(self.root_path, split))

        self.h5_folder = h5_folder

        self.transform3d = transform3d

        self.range01 = range01
        self.pos_int16 = pos_int16
        self.random_crop = random_crop
        self.factor_rand = factor_rand
        self.factor = factor

    def __getitem__(self, index):
        h5_file = h5py.File(os.path.join(self.h5_path, self.split[index]+".h5"), 'r')
        cls = int(np.asarray((h5_file["label"])))
        P = np.asarray(h5_file["points"])

        if len(P.shape) == 3:
            P = P.reshape(-1, 3)

        if self.pos_int16:
            P = (P/1000).astype(np.float32)

        F = None
        if 'feat' in h5_file.keys():
            F = np.asarray(h5_file["feat"], dtype=np.float32)
            if len(F.shape) == 1:
                F = np.transpose([F])
            elif len(F.shape) == 3:
                F = F.reshape(-1, 3)

            if self.range01:
                F = F/255
        else:
            raise RuntimeError('node feat do not exist')

        if self.random_crop:
            if self.factor_rand is True:
                factor = np.random.randint(low=self.factor*100, high=100, dtype=np.int)/100
            else:
                factor = self.factor
            P, F = random_crop_3D(P, F, factor)

        if self.transform3d is not None:
            if self.transform3d["dropout"] > 0:
                P, F = dropout(P, F, self.transform3d["dropout"])
            M = np.eye(3)
            if self.transform3d["rot"]:
                angle = random.uniform(0, 2*math.pi)
                M = np.dot(transforms3d.axangles.axangle2mat([0, 1, 0], angle), M)

            if self.transform3d["mirror"] > 0:
                if random.random() < self.transform3d["mirror"]/2:
                    M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), M)
                if random.random() < self.transform3d["mirror"]/2:
                    M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), M)
            P = np.dot(P, M.T)
        P -= np.min(P, axis=0)

        F = torch.tensor(F)
        P = torch.tensor(P)

        data = Data(x=F, pos=P, y=cls, c=index)

        return data

    def __len__(self):
        return len(self.split)
