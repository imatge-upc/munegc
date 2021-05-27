"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import torch
import torch_geometric
import Fusion2D3DMUNEGC.geometric3d.graph_model as models
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_scatter import scatter
from torch_cluster import nearest

class MultiModalGroupFusion(torch.nn.Module):
    def __init__(self, pool_rad):
        super(MultiModalGroupFusion, self).__init__()
        self.pool_rad = pool_rad

    def forward(self, b1, b2):
        pos = torch.cat([b1.pos, b2.pos], 0)
        batch = torch.cat([b1.batch, b2.batch], 0)

        batch, sorted_indx = torch.sort(batch)
        inv_indx = torch.argsort(sorted_indx)
        pos = pos[sorted_indx, :]

        start = pos.min(dim=0)[0] - self.pool_rad * 0.5
        end = pos.max(dim=0)[0] + self.pool_rad * 0.5

        cluster = torch_geometric.nn.voxel_grid(pos, batch, self.pool_rad, start=start, end=end)
        cluster, perm = consecutive_cluster(cluster)

        superpoint = scatter(pos, cluster, dim=0, reduce='mean')
        new_batch = batch[perm]

        cluster = nearest(pos, superpoint, batch, new_batch)

        cluster, perm = consecutive_cluster(cluster)

        pos = scatter(pos, cluster, dim=0, reduce='mean')
        branch_mask = torch.zeros(batch.size(0)).bool()
        branch_mask[0:b1.batch.size(0)] = 1

        cluster = cluster[inv_indx]

        nVoxels = len(cluster.unique())

        x_b1 = torch.ones(nVoxels, b1.x.shape[1], device=b1.x.device)
        x_b2 = torch.ones(nVoxels, b2.x.shape[1], device=b2.x.device)

        x_b1 = scatter(b1.x, cluster[branch_mask], dim=0, out=x_b1, reduce='mean')
        x_b2 = scatter(b2.x, cluster[~branch_mask], dim=0, out=x_b2, reduce='mean')

        x = torch.cat([x_b1, x_b2], 1)

        batch = batch[perm]

        b1.x = x
        b1.pos = pos
        b1.batch = batch
        b1.edge_attr = None
        b1.edge_index = None

        return b1

    def extra_repr(self):
        s = 'pool_rad: {pool_rad}'
        return s.format(**self.__dict__)


class TwoStreamNetwork(torch.nn.Module):
    def __init__(self, graph_net_conf,
                 features_b1, features_b2, rad_fuse_pool,
                 multigpu,
                 features_proj_b1=64,
                 features_proj_b2=64,
                 proj_b1=True,
                 proj_b2=True):

        super(TwoStreamNetwork, self).__init__()
        self.rad_fuse_pool = float(rad_fuse_pool)
        if proj_b1:
            self.proj_b1 = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Conv1d(
                features_b1, features_proj_b1, kernel_size=1, bias=False))

            features_b1 = features_proj_b1
        else:
            self.proj_b1 = None

        if proj_b2:
            self.proj_b2 = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Conv1d(
                features_b2, features_proj_b2, kernel_size=1, bias=False))
            features_b2 = features_proj_b2
        else:
            self.proj_b2 = None

        self.multimodal_gp_fusion = MultiModalGroupFusion(rad_fuse_pool)
        self.class_network = models.GraphNetwork(graph_net_conf, features_b1 + features_b2, multigpu)

    def forward(self, data_b1, data_b2):

        batch_b1 = torch_geometric.utils.to_dense_batch(data_b1.x, data_b1.batch)
        batch_b2 = torch_geometric.utils.to_dense_batch(data_b2.x, data_b2.batch)

        x_b1 = batch_b1[0].permute(0, 2, 1)
        x_b2 = batch_b2[0].permute(0, 2, 1)
        if self.proj_b1 is not None:
            x_b1 = self.proj_b1(x_b1)
        if self.proj_b2 is not None:
            x_b2 = self.proj_b2(x_b2)

        data_b1.x = x_b1.permute(0, 2, 1).reshape(-1, x_b1.size(1))[batch_b1[1].view(-1)]
        data_b2.x = x_b2.permute(0, 2, 1).reshape(-1, x_b2.size(1))[batch_b2[1].view(-1)]

        data = self.multimodal_gp_fusion(data_b1, data_b2)
        return self.class_network(data)
