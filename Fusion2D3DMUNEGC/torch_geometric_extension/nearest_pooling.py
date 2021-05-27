"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import torch

import torch_geometric.nn as nn_geometric
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_scatter import scatter
from torch_cluster import nearest
from .graph_reg import GraphReg


class NVPooling(torch.nn.Module):
    def __init__(self, pool_rad, aggr='max'):
        super(NVPooling, self).__init__()
        self.pool_rad = pool_rad

        self.aggr = aggr.strip().lower()
        if self.aggr == 'max':
            self._aggr = 'max'
        elif self.aggr == 'avg':
            self._aggr = 'mean'
        else:
            raise RuntimeError("Invalid aggregation method in Graph Pooling Layer")

    def forward(self, data):

        cluster = nn_geometric.voxel_grid(data.pos, data.batch, self.pool_rad,
                                          start=data.pos.min(dim=0)[0] - self.pool_rad * 0.5,
                                          end=data.pos.max(dim=0)[0] + self.pool_rad * 0.5)
        cluster, perm = consecutive_cluster(cluster)

        new_pos = scatter(data.pos, cluster, dim=0, reduce='mean')
        new_batch = data.batch[perm]

        cluster = nearest(data.pos, new_pos, data.batch, new_batch)

        cluster, perm = consecutive_cluster(cluster)

        data.x = scatter(data.x, cluster, dim=0, reduce=self._aggr)
        data.pos = scatter(data.pos, cluster, dim=0, reduce='mean')

        data.batch = data.batch[perm]
        data.edge_index = None
        data.edge_attr = None

        return data

    def extra_repr(self):
        s = 'aggr={aggr}'
        s += ', pool_rad={pool_rad}'
        return s.format(**self.__dict__)


class NVGraphPooling(torch.nn.Module):
    def __init__(self, pool_rad, aggr='max',
                 neighs=0.9, rad_neigh=None, self_loop=True,
                 edge_attr=None, flow='source_to_target'):

        super(NVGraphPooling, self).__init__()
        self.pool_rad = pool_rad
        self.aggr = aggr.strip().lower()
        if self.aggr == 'max':
            self._aggr = 'max'
        elif self.aggr == 'avg':
            self._aggr = 'mean'
        else:
            raise RuntimeError("Invalid aggregation method in Graph Pooling Layer")

        self.n_neighs = neighs
        self.rad_neigh = rad_neigh

        self.edge_attr = edge_attr

        self.knn = True if (rad_neigh is None) else False

        self.flow = flow

        self.graph_reg = GraphReg(neighs, rad_neigh, self.knn, self_loop, edge_attr, flow=flow)

    def forward(self, data):

        cluster = nn_geometric.voxel_grid(data.pos, data.batch, self.pool_rad,
                                          start=data.pos.min(dim=0)[0] - self.pool_rad * 0.5,
                                          end=data.pos.max(dim=0)[0] + self.pool_rad * 0.5)

        cluster, perm = consecutive_cluster(cluster)

        new_pos = scatter(data.pos, cluster, dim=0, reduce='mean')
        new_batch = data.batch[perm]

        cluster = nearest(data.pos, new_pos, data.batch, new_batch)

        cluster, perm = consecutive_cluster(cluster)

        data.x = scatter(data.x, cluster, dim=0, reduce=self._aggr)
        data.pos = scatter(data.pos, cluster, dim=0, reduce='mean')

        data.batch = data.batch[perm]

        data.edge_index = None
        data.edge_attr = None

        data = self.graph_reg(data)

        return data

    def extra_repr(self):
        s = 'aggr={aggr}'
        s += ', pool_rad={pool_rad}'
        if self.knn:
            s += ', knn={n_neighs}'
        else:
            s += ', num_neighs={n_neighs}, rad_neigh={rad_neigh}'

        return s.format(**self.__dict__)
