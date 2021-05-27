"""
	2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import torch
from torch.nn import Parameter
import torch.nn.init as init
import math
from torch_scatter import scatter

def create_fnet(widths, nfeat, nfeato, llbias=True, tanh_activation=True):
    fnet_modules = []
    for k in range(len(widths)-1):
        fnet_modules.append(torch.nn.Linear(widths[k], widths[k+1]))
        init.orthogonal_(fnet_modules[-1].weight, gain=init.calculate_gain('relu'))
        fnet_modules.append(torch.nn.ReLU(True))
    fnet_modules.append(torch.nn.Linear(widths[-1], nfeat*nfeato, bias=llbias))
    init.orthogonal_(fnet_modules[-1].weight)
    if tanh_activation:
        fnet_modules.append(torch.nn.Tanh())
    return torch.nn.Sequential(*fnet_modules)


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def create_agc(nfeat, nfeato, fnet_widths, fnet_llbias=True, fnet_tanh=False,
               bias=False, flow='source_to_target'):

    fnet = create_fnet(fnet_widths, nfeat, nfeato, llbias=fnet_llbias, tanh_activation=fnet_tanh)

    return AGC(nfeat, nfeato, fnet, aggr="mean", flow=flow, bias=bias)


class AGC(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 nn,
                 aggr='mean',
                 flow='source_to_target',
                 bias=False,
                 eps=1e-9):

        super(AGC, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self._wg = nn
        self.aggr = aggr
        assert flow in ['source_to_target', 'target_to_source']
        self.flow = flow

        self.eps = eps
        self.bias_bool = bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr

        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)

        src_indx = edge_index[j]
        target_indx = edge_index[i]

        # weights computation
        out = self._wg(edge_attr)
        out = out.view(-1, self.in_channels, self.out_channels)

        N, C = x.size()

        feat = x[src_indx]

        out = torch.matmul(feat.unsqueeze(1), out).squeeze(1)
        out = scatter(out, target_indx, dim=0, dim_size=N, reduce=self.aggr)

        if self.bias is not None:
            out = out + self.bias

        return out

    def extra_repr(self):
        s = '{}({}, {}'.format(self.__class__.__name__, self.in_channels,
                               self.out_channels)
        s += ', Bias: {}'.format(self.bias_bool)
        s += ')'
        return s.format(**self.__dict__)
