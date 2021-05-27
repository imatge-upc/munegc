"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import torch
import torch_geometric.nn as nn_geometric


class GlobalPCPooling(torch.nn.Module):
    def __init__(self, aggr):
        super(GlobalPCPooling, self).__init__()
        self.aggr = aggr
        if self.aggr == 'max':
            self.pool = nn_geometric.global_max_pool
        elif self.aggr == 'avg':
            self.pool = nn_geometric.global_mean_pool
        else:
            raise RuntimeError("Invalid aggration method in Global Graph Pooling Layer")

    def forward(self, x, batch):
        return self.pool(x, batch)

    def extra_repr(self):
        s = 'aggr={aggr}'
        return s.format(**self.__dict__)
