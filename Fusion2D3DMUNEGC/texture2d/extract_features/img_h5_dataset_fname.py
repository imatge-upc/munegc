"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import torch
import numpy as np
import os
import h5py
from Fusion2D3DMUNEGC.utilities import utils


#
# Modified version of ImageH5Dataset that returns the name of the file of each sample.
#

class ImageH5Dataset(torch.utils.data.Dataset):
    def __init__(self, root_path, h5_folder, split,
                 transform=None, range01=False):

        self.root_path = root_path

        self.h5_path = os.path.join(self.root_path, h5_folder)
        self.split = utils.read_string_list(os.path.join(self.root_path, split))

        self.h5_folder = h5_folder

        self.transform = transform

        self.range01 = range01
    def __getitem__(self, index):
        h5_file = h5py.File(os.path.join(self.h5_path, self.split[index]+".h5"), 'r')
        scene = int(np.asarray((h5_file["label"])))
        img = torch.tensor(np.asarray(h5_file["img"]), dtype=torch.float).permute(2, 0, 1)
        if self.range01:
            img = img/255
        if self.transform:
            img = self.transform(img)

        return img, scene, self.split[index]

    def __len__(self):
        return len(self.split)
