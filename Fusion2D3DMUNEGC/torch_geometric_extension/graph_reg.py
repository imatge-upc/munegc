"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import torch
import torch_geometric.transforms as T

from .spherical import Spherical
from .cartesian import Cartesian
from .feature_distances import FeatureDistances


def numberEdgeAttr(edge_attr, nfeat):

    nEA = 0
    if edge_attr is not None:
        if type(edge_attr) == str:
            edge_attr = [edge_attr]
        for attr in edge_attr:
            attr = attr.strip().lower()
            if attr == 'poscart':
                nEA = nEA + 3

            elif attr == 'posspherical':
                nEA = nEA + 3

            elif attr == 'featureoffsets':
                nEA = nEA + nfeat

            elif attr == 'featurel2':
                nEA = nEA + 1

            else:
                raise RuntimeError('{} is not supported'.format(attr))

    return nEA


class GraphReg(torch.nn.Module):
    def __init__(self, n_neigh=9, rad_neigh=0.1, knn=None, self_loop=True,
                 edge_attr=None, flow='source_to_target'):
        super(GraphReg, self).__init__()
        # defining graph transform
        graph_transform_list = []
        self.del_edge_attr = False
        self.knn = knn
        self.n_neigh = n_neigh
        self.rad_neigh = rad_neigh
        self.self_loop = self_loop
        self.edge_attr = edge_attr
        if self.knn == True:
            graph_transform_list.append(T.KNNGraph(n_neigh, loop=self_loop,
                                                   flow=flow))

        elif self.knn == False:
            graph_transform_list.append(T.RadiusGraph(self.rad_neigh, loop=self_loop,
                                                      max_num_neighbors=n_neigh,
                                                      flow=flow))
        else:
            print("Connectivity of the graph will not be re-generated")

        # edge attr
        if edge_attr is not None:
            self.del_edge_attr = True
            if type(edge_attr) == str:
                if edge_attr:
                    edge_attr = [attr.strip() for attr in edge_attr.split('-')]
                else:
                    edge_attr = []
            for attr in edge_attr:
                attr = attr.strip().lower()

                if attr == 'poscart':
                    graph_transform_list.append(Cartesian(norm=False, cat=True))

                elif attr == 'posspherical':
                    graph_transform_list.append(Spherical(cat=True))

                elif attr == 'featureoffsets':
                    graph_transform_list.append(FeatureDistances(metric='offset', cat=True))

                elif attr == 'featurel2':
                    graph_transform_list.append(FeatureDistances(metric='l2', cat=True))

                else:
                    raise RuntimeError('{} is not supported'.format(attr))
        self.graph_transform = T.Compose(graph_transform_list)

    def forward(self, data):
        if self.del_edge_attr:
            data.edge_attr = None
        data = self.graph_transform(data)
        return data

    def extra_repr(self):
        s = "knn={knn}"
        if self.knn == True:
            s += ", n_neigh={n_neigh}"
            s += ", self_loop={self_loop}"
        elif self.knn == False:
            s += ", n_neigh={n_neigh}"
            s += ", rad_neigh={rad_neigh}"
            s += ", self_loop={self_loop}"

        s += ", edge_attr={edge_attr}"
        return s.format(**self.__dict__)
