"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import torch
import torch_geometric

from .agc import create_agc
from .graph_reg import GraphReg
from .graph_reg import numberEdgeAttr


class MUNEGC(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 neighs=9, rad_neigh=None,
                 fnetw=[128], edge_attr=['posspherical'], edge_attr_feat=None,
                 fnet_llbias=True, fnet_tanh=True,
                 aggr='avg', bias=False, flow='source_to_target'):

        super(MUNEGC, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_neighs = neighs
        self.rad_neigh = rad_neigh

        self.edge_attr = edge_attr
        self.edge_attr_feat = edge_attr_feat if (edge_attr_feat is not None) else edge_attr
        self.flow = flow
        self.aggr = aggr.strip().lower()

        self.knn = True if (rad_neigh is None) else False

        self.fnetw_geo = [numberEdgeAttr(self.edge_attr, self.in_channels)] + fnetw
        self.fnetw_feat = [numberEdgeAttr(self.edge_attr_feat, self.in_channels)] + fnetw

        self.agc_geo = create_agc(self.in_channels, self.out_channels,
                                  self.fnetw_geo,
                                  fnet_llbias=fnet_llbias,
                                  fnet_tanh=fnet_tanh, bias=bias, flow=flow)

        self.agc_feat = create_agc(self.in_channels, self.out_channels,
                                   self.fnetw_feat,
                                   fnet_llbias=fnet_llbias,
                                   fnet_tanh=fnet_tanh, bias=bias, flow=flow)

        self.graph_gen_geo = GraphReg(n_neigh=neighs, rad_neigh=rad_neigh, knn=self.knn,
                                      self_loop=True, edge_attr=self.edge_attr,
                                      flow=flow)

        self.gen_edge_attr_feat = GraphReg(knn=None, edge_attr=self.edge_attr_feat,
                                       flow=flow)

    def forward(self, data):
        data = self.graph_gen_geo(data)
        x_geo = self.agc_geo(data.x, data.edge_index, data.edge_attr.float())

        data.edge_index = None
        data.edge_attr = None

        data.edge_index = torch_geometric.nn.knn_graph(data.x, self.n_neighs,
                                                       data.batch, loop=True,
                                                       flow=self.flow)
        data = self.gen_edge_attr_feat(data)

        x_feat = self.agc_feat(data.x, data.edge_index, data.edge_attr.float())

        if self.aggr == 'avg':
            data.x = (x_geo + x_feat)/2

        elif self.aggr == 'max':
            data.x = torch.max(x_geo, x_feat).squeeze(-1)
        else:
            raise RuntimeError('Invalid aggregation')
        data.edge_index = None
        data.edge_attr = None

        return data

    def extra_repr(self):
        s = '{}({}, {}'.format(self.__class__.__name__, self.in_channels,
                               self.out_channels)
        s += ', aggr: {}'.format(self.aggr)
        s += ')'
        return s.format(**self.__dict__)
