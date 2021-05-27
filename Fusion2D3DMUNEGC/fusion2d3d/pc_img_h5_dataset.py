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
from Fusion2D3DMUNEGC.utilities import utils
from torch_geometric.data import Batch


def custom_collate(data_list):
    batch_1 = Batch.from_data_list([d[0] for d in data_list])
    batch_2 = Batch.from_data_list([d[1] for d in data_list])
    return batch_1, batch_2


def dropout(P, F, p):
    idx = random.sample(range(P.shape[0]), int(math.ceil((1-p)*P.shape[0])))
    return P[idx, :], F[idx, :] if F is not None else None


class PCIMGH5Dataset(torch.utils.data.Dataset):
    def __init__(self, root_path, h5_folder_b1, h5_folder_b2, split,
                 transform3d=None, pos_int16=False):

        self.root_path = root_path

        self.h5_path_b1 = os.path.join(self.root_path, h5_folder_b1)
        self.h5_path_b2 = os.path.join(self.root_path, h5_folder_b2)

        self.split = utils.read_string_list(os.path.join(self.root_path, split))

        self.h5_folder_b1 = h5_folder_b1
        self.h5_folder_b2 = h5_folder_b2

        self.transform3d = transform3d

        self.pos_int16 = pos_int16

    def __getitem__(self, index):
        h5_file_b1 = h5py.File(os.path.join(self.h5_path_b1, self.split[index]+".h5"), 'r')
        h5_file_b2 = h5py.File(os.path.join(self.h5_path_b2, self.split[index]+".h5"), 'r')

        cls_b1 = int(np.asarray((h5_file_b1["label"])))
        cls_b2 = int(np.asarray((h5_file_b2["label"])))

        if cls_b1 != cls_b2:
            raise RuntimeError("Branches have different classes")

        P_b1 = np.asarray(h5_file_b1["points"])
        P_b2 = np.asarray(h5_file_b2["points"])
        if self.pos_int16:
            P_b1 = (P_b1/1000).astype(np.float32)
            P_b2 = (P_b2/1000).astype(np.float32)

        F_b1 = None
        F_b2 = None

        if 'feat' in h5_file_b1.keys():
            F_b1 = np.asarray(h5_file_b1["feat"], dtype=np.float32)
            if len(F_b1.shape) == 1:
                F_b1 = np.transpose([F_b1])
        else:
            raise RuntimeError('node feat do not exist in branch 1')

        if 'feat' in h5_file_b2.keys():
            F_b2 = np.asarray(h5_file_b2["feat"], dtype=np.float32)
            if len(F_b2.shape) == 1:
                F_b2 = np.transpose([F_b2])
        else:
            raise RuntimeError('node feat do not exist in branch 2')

        if self.transform3d is not None:
            if self.transform3d["dropout"] > 0:
                P_b1, F_b1 = dropout(P_b1, F_b1, self.transform3d["dropout"])
                P_b2, F_b2 = dropout(P_b2, F_b2, self.transform3d["dropout"])
            M = np.eye(3)
            if self.transform3d["rot"]:
                angle = random.uniform(0, 2*math.pi)
                M = np.dot(transforms3d.axangles.axangle2mat([0, 1, 0], angle), M)

            if self.transform3d["mirror"] > 0:
                if random.random() < self.transform3d["mirror"]/2:
                    M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), M)
                if random.random() < self.transform3d["mirror"]/2:
                    M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), M)
            P_b1 = np.dot(P_b1, M.T)
            P_b2 = np.dot(P_b2, M.T)

        P_b1 -= np.min(P_b1, axis=0)
        P_b2 -= np.min(P_b2, axis=0)

        data_1 = Data(x=torch.tensor(F_b1), pos=torch.tensor(P_b1), y=cls_b1)
        data_2 = Data(x=torch.tensor(F_b2), pos=torch.tensor(P_b2), y=cls_b2)

        return data_1, data_2

    def __len__(self):
        return len(self.split)
