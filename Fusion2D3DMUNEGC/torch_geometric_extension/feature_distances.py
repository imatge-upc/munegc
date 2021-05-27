"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import torch


class FeatureDistances(object):
    r"""Saves the relative feature distance of linked nodes in its edge
    attributes.
    Args:
        metric (string, optional): the type of distance. Default offsets.
                The following metrics can be calculated: offset, l2,
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
    """

    def __init__(self, metric='offset', cat=True):
        self.cat = cat
        self.metric = metric.strip().lower()

    def __call__(self, data):
        (row, col), feat, pseudo = data.edge_index, data.x, data.edge_attr

        feat_offsets = feat[col] - feat[row]
        feat_offsets = feat_offsets.view(-1, 1) if feat_offsets.dim() == 1 else feat_offsets
        if self.metric == 'offset':
            distance = feat_offsets
        elif self.metric == 'l2':
            distance = torch.norm(feat_offsets, p=2, dim=-1).unsqueeze(-1)
        else:
            raise RuntimeError("This feature metric is not implemented")

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, distance.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = distance

        return data

    def __repr__(self):
        return '{}(norm={})'.format(self.__class__.__name__,
                                                  self.metric)
