"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

from math import pi as PI
import torch


class Spherical(object):
    r"""Saves the relative spherical coordinates of linked nodes in its edge
    attributes.
    Args:
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
    """

    def __init__(self, cat=True):
        self.cat = cat

    def __call__(self, data):

        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr
        assert pos.dim() == 2 and pos.size(1) == 3

        cart = pos[col] - pos[row]

        rho = torch.norm(cart, p=2, dim=-1)

        theta = torch.atan2(cart[..., 1], cart[..., 0]).view(-1, 1)
        theta = theta + (theta < 0).type_as(theta) * (2 * PI)

        phi = torch.acos(cart[..., 2] / (rho + 1e-6)).view(-1, 1)

        rho = rho.view(-1, 1)
        spher = torch.cat([rho, theta, phi], dim=-1)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, spher.type_as(pos)], dim=-1)
        else:
            data.edge_attr = spher

        return data
