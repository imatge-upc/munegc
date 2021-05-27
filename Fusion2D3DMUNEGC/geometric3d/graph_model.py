"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import torch
import torch_geometric
from Fusion2D3DMUNEGC import torch_geometric_extension as ext
from Fusion2D3DMUNEGC.utilities.utils import gpuid2device

import numpy as np
import ast


class GraphNetwork(torch.nn.Module):
    def __init__(self, config, nfeat, multigpu=False, default_fnet_widths=[128],
                 default_fnet_llbias=False, default_fnet_tanh=False,
                 default_edge_attr='poscart', default_conv_bias=False):
        super(GraphNetwork, self).__init__()

        self.multigpu = multigpu
        self.devices = []
        self.flow = 'source_to_target'
        nfeat = nfeat
        nEdgeAttr = 0
        self.intermediate = None

        for d, conf in enumerate(config.split(',')):
            fnet_widths = default_fnet_widths
            conf = conf.strip().split('_')
            device = None
            if default_edge_attr is not None:
                edge_attr = [attr.strip() for attr in default_edge_attr.split('-')]
            else:
                edge_attr = []
            fnet_tanh = default_fnet_tanh
            conv_bias = default_conv_bias
            fnet_llbias = default_fnet_llbias

            # Graph Generation
            if conf[0] == 'ggknn':
                if len(conf) < 2:
                    raise RuntimeError("{} Graph Generation layer requires more arguments".format(d))
                neigh = int(conf[1])
                if len(conf) > 2:
                    if conf[2].isdigit():
                        device = conf[2]
                    else:
                        edge_attr = [attr.strip() for attr in conf[2].split('-')]
                        if len(conf) == 4:
                            device = conf[3]
                        elif len(conf) > 4:
                            raise RuntimeError("Invalid parameters in {} ggknn layer".format(d))

                module = ext.GraphReg(knn=True, n_neigh=neigh, edge_attr=edge_attr, self_loop=True, flow=self.flow)
                nEdgeAttr = ext.numberEdgeAttr(edge_attr, nfeat)

            elif conf[0] == 'ggrad':
                if len(conf) < 3:
                    raise RuntimeError("{} Graph Generation layer requires more arguments".format(d))
                rad = float(conf[1])
                neigh = int(conf[2])
                if len(conf) > 3:
                    if conf[3].isdigit():
                        device = conf[3]
                    else:
                        edge_attr = [attr.strip() for attr in conf[3].split('-')]
                        if len(conf) == 5:
                            device = conf[4]
                        elif len(conf) > 5:
                            raise RuntimeError("Invalid parameters in {} ggrad layer".format(d))

                module = ext.GraphReg(knn=False, n_neigh=neigh, rad_neigh=rad,
                                      edge_attr=edge_attr, self_loop=True, flow=self.flow)
                nEdgeAttr = ext.numberEdgeAttr(edge_attr, nfeat)

            elif conf[0] == 'f':
                if len(conf) < 2:
                    raise RuntimeError("{} Fully connected layer requires as argument the output features".format(d))
                nfeato = int(conf[1])
                module = torch.nn.Linear(nfeat, nfeato)
                nfeat = nfeato
                if len(conf) == 3:
                    if conf[2].isdigit():
                        device = conf[2]
                    elif conf[2] == 'cp':
                        torch.nn.init.constant_(module.bias, -np.log(nfeato-1))
                if len(conf) == 4:
                    device = conf[3]
                elif len(conf) > 4:
                    raise RuntimeError("Invalid parameters in {} fully connected layer".format(d))

            elif conf[0] == 'b':
                module = torch.nn.BatchNorm1d(nfeat, affine=True, track_running_stats=True)
                if len(conf) == 2:
                    device = conf[1]
                elif len(conf) > 3:
                    raise RuntimeError("Invalid parameters in {} batchnom layer".format(d))

            elif conf[0] == 'r':
                module = torch.nn.ReLU(True)
                if len(conf) == 2:
                    device = conf[1]
                elif len(conf) > 3:
                    raise RuntimeError("Invalid parameters in {} relu layer".format(d))
            elif conf[0] == 'd':
                if len(conf) < 2:
                    raise RuntimeError(
                        "{} Dropout layer requires as argument the probabity to zeroed an element".format(d))
                prob = float(conf[1])
                module = torch.nn.Dropout(prob, inplace=False)
                if len(conf) == 3:
                    device = conf[2]
                elif len(conf) > 3:
                    raise RuntimeError("Invalid parameters in {} dropout layer".format(d))

            elif conf[0] == 'agc':
                if len(conf) < 2:
                    raise RuntimeError("{} agc layer requires as argument the output features".format(d))
                nfeato = int(conf[1])
                if len(conf) > 2:
                    if conf[2].isdigit():
                        device = conf[2]
                    else:
                        params = [param.strip() for param in conf[2].split('-')]
                        for param in params:
                            p = [p.strip() for p in param.split(':')]
                            param = p[0]
                            if param == 'bias':
                                conv_bias = int(p[1])
                            elif param == 'fwidths':
                                fnet_widths = ast.literal_eval(p[1].replace('#', ','))
                            elif param == 'ftanh':
                                fnet_tanh = int(p[1])

                        if len(conf) == 4:
                            device = conf[3]

                        elif len(conf) > 4:
                            raise RuntimeError("Invalid parameters in {} agc layer".format(d))

                module = ext.create_agc(nfeat, nfeato, [nEdgeAttr] + fnet_widths,
                                        fnet_llbias=fnet_llbias,
                                        bias=conv_bias,
                                        fnet_tanh=fnet_tanh,
                                        flow=self.flow)
                nfeat = nfeato

            elif conf[0] == 'multigraphconvknnfeat':
                if len(conf) < 3:
                    raise RuntimeError("{} MUNEGC layer requires as argument the output features".format(d))
                n_neigh = int(conf[1])
                nfeato = int(conf[2])
                if len(conf) > 3:
                    if conf[3].isdigit():
                        device = conf[3]
                    else:
                        params = [param.strip() for param in conf[3].split('-')]
                        for param in params:
                            p = [p.strip() for p in param.split(':')]
                            param = p[0]
                            if param == 'bias':
                                conv_bias = int(p[1])
                            elif param == 'fwidths':
                                fnet_widths = ast.literal_eval(p[1].replace('#', ','))
                            elif param == 'ftanh':
                                fnet_tanh = int(p[1])

                        if len(conf) == 5:
                            device = conf[4]

                        elif len(conf) > 5:
                            raise RuntimeError("Invalid parameters in {} MUNEGC layer".format(d))

                module = ext.MUNEGC(nfeat, nfeato,
                                    neighs=n_neigh,
                                    fnetw=fnet_widths,
                                    edge_attr=['posspherical'],
                                    edge_attr_feat=['featureoffsets'],
                                    fnet_llbias=fnet_llbias,
                                    fnet_tanh=fnet_tanh,
                                    bias=conv_bias,
                                    aggr='avg',
                                    flow=self.flow)

                nfeat = nfeato

            # MultiGraphConvolution
            elif conf[0] == 'multigraphconv':
                if len(conf) < 3:
                    raise RuntimeError("{} MUNEGC layer requires as argument the output features".format(d))
                n_neigh = int(conf[1])
                nfeato = int(conf[2])
                if len(conf) > 3:
                    if conf[3].isdigit():
                        device = conf[3]
                    else:
                        params = [param.strip() for param in conf[3].split('-')]
                        for param in params:
                            p = [p.strip() for p in param.split(':')]
                            param = p[0]
                            if param == 'bias':
                                conv_bias = int(p[1])
                            elif param == 'fwidths':
                                fnet_widths = ast.literal_eval(p[1].replace('#', ','))
                            elif param == 'ftanh':
                                fnet_tanh = int(p[1])

                        if len(conf) == 5:
                            device = conf[4]

                        elif len(conf) > 5:
                            raise RuntimeError("Invalid parameters in {} MUNEGC layer".format(d))

                module = ext.MUNEGC(nfeat, nfeato,
                                    neighs=n_neigh,
                                    fnetw=fnet_widths,
                                    edge_attr=edge_attr,
                                    fnet_llbias=fnet_llbias,
                                    fnet_tanh=fnet_tanh,
                                    bias=conv_bias,
                                    aggr='avg',
                                    flow=self.flow)

                nfeat = nfeato

            # MultiGraphConvolution
            elif conf[0] == 'multigraphconvmax':
                if len(conf) < 3:
                    raise RuntimeError("{} MUNEGC layer requires as argument the output features".format(d))
                n_neigh = int(conf[1])
                nfeato = int(conf[2])
                if len(conf) > 3:
                    if conf[3].isdigit():
                        device = conf[3]
                    else:
                        params = [param.strip() for param in conf[3].split('-')]
                        for param in params:
                            p = [p.strip() for p in param.split(':')]
                            param = p[0]
                            if param == 'bias':
                                conv_bias = int(p[1])
                            elif param == 'fwidths':
                                fnet_widths = ast.literal_eval(p[1].replace('#', ','))
                            elif param == 'ftanh':
                                fnet_tanh = int(p[1])

                        if len(conf) == 5:
                            device = conf[4]

                        elif len(conf) > 5:
                            raise RuntimeError("Invalid parameters in {} MUNEGC layer".format(d))

                module = ext.MUNEGC(nfeat, nfeato,
                                    neighs=n_neigh,
                                    fnetw=fnet_widths,
                                    edge_attr=edge_attr,
                                    fnet_llbias=fnet_llbias,
                                    fnet_tanh=fnet_tanh,
                                    bias=conv_bias,
                                    aggr='max',
                                    flow=self.flow)
                nfeat = nfeato

            # MultiGraphConvolutionGen
            elif conf[0] == 'multigraphconvradbasedgen':
                if len(conf) < 5:
                    raise RuntimeError("{} MUNEGC layer requires as argument the output features".format(d))
                n_neigh = int(conf[1])
                rad_neigh = float(conf[2])
                nfeato = int(conf[4])
                aggr = str(conf[3])
                if len(conf) > 5:
                    if conf[5].isdigit():
                        device = conf[5]
                    else:
                        params = [param.strip() for param in conf[5].split('-')]
                        for param in params:
                            p = [p.strip() for p in param.split(':')]
                            param = p[0]
                            if param == 'bias':
                                conv_bias = int(p[1])
                            elif param == 'fwidths':
                                fnet_widths = ast.literal_eval(p[1].replace('#', ','))
                            elif param == 'ftanh':
                                fnet_tanh = int(p[1])

                        if len(conf) == 7:
                            device = conf[6]

                        elif len(conf) > 7:
                            raise RuntimeError("Invalid parameters in {} MUNEGC layer".format(d))

                module = ext.MUNEGC(nfeat, nfeato,
                                    neighs=n_neigh,
                                    rad_neigh=rad_neigh,
                                    fnetw=fnet_widths,
                                    edge_attr=edge_attr,
                                    fnet_llbias=fnet_llbias,
                                    fnet_tanh=fnet_tanh,
                                    bias=conv_bias,
                                    aggr=aggr,
                                    flow=self.flow)
                nfeat = nfeato

            elif conf[0] == 'pvknn':
                if len(conf) < 4:
                    raise RuntimeError("{} Pool layer requires more arguments".format(d))
                aggr = conf[1]
                pradius = float(conf[2])
                nn = int(conf[3])
                if len(conf) > 4:
                    if conf[4].isdigit():
                        device = conf[4]
                    else:
                        edge_attr = [attr.strip() for attr in conf[4].split('-')]
                        if len(conf) == 6:
                            device = conf[5]
                        elif len(conf) > 6:
                            raise RuntimeError("Invalid parameters in {} pool layer".format(d))

                module = ext.VGraphPooling(pradius, aggr=aggr,
                                           neighs=nn, self_loop=True,
                                           edge_attr=edge_attr,
                                           flow=self.flow)

                nEdgeAttr = ext.numberEdgeAttr(edge_attr, nfeat)

            elif conf[0] == 'pvrnn':
                if len(conf) < 5:
                    raise RuntimeError("{} Pool layer requires more arguments".format(d))
                aggr = conf[1]
                pradius = float(conf[2])
                rad_neigh = float(conf[3])
                nn = int(conf[4])

                if len(conf) > 5:
                    if conf[5].isdigit():
                        device = conf[5]
                    else:
                        edge_attr = [attr.strip() for attr in conf[5].split('-')]
                        if len(conf) == 7:
                            device = conf[6]
                        elif len(conf) > 7:
                            raise RuntimeError("Invalid parameters in {} pool layer".format(d))

                module = ext.VGraphPooling(pradius, aggr=aggr,
                                            neighs=nn, rad_neigh=rad_neigh,
                                            self_loop=True, edge_attr=edge_attr,
                                            flow=self.flow)

                nEdgeAttr = ext.numberEdgeAttr(edge_attr, nfeat)

            # voxel  pooling
            elif conf[0] == 'pv':
                if len(conf) < 3:
                    raise RuntimeError("{} Pool layer requires more arguments".format(d))
                aggr = conf[1]
                pradius = float(conf[2])
                if len(conf) == 4:
                    if conf[3].isdigit():
                        device = conf[3]
                module = ext.VPooling(pool_rad=pradius, aggr=aggr)

            # nearest voxel pooling

            elif conf[0] == 'pnv':
                if len(conf) < 3:
                    raise RuntimeError("{} Pool layer requires more arguments".format(d))
                aggr = conf[1]
                pradius = float(conf[2])
                if len(conf) == 4:
                    if conf[3].isdigit():
                        device = conf[3]
                module = ext.NVPooling(pool_rad=pradius, aggr=aggr)

            # KNN pooling layer nearest voxel
            elif conf[0] == 'pnvknn':
                if len(conf) < 4:
                    raise RuntimeError("{} Pool layer requires more arguments".format(d))
                aggr = conf[1]
                pradius = float(conf[2])
                nn = int(conf[3])
                if len(conf) > 4:
                    if conf[4].isdigit():
                        device = conf[4]
                    else:
                        edge_attr = [attr.strip() for attr in conf[4].split('-')]
                        if len(conf) == 6:
                            device = conf[5]
                        elif len(conf) > 6:
                            raise RuntimeError("Invalid parameters in {} pool layer".format(d))

                module = ext.NVGraphPooling(pradius, aggr=aggr,
                                            neighs=nn, self_loop=True,
                                            edge_attr=edge_attr,
                                            flow=self.flow)

                nEdgeAttr = ext.numberEdgeAttr(edge_attr, nfeat)

            # Radius pooling layer nearest voxel
            elif conf[0] == 'pnvrnn':
                if len(conf) < 5:
                    raise RuntimeError("{} Pool layer requires more arguments".format(d))
                aggr = conf[1]
                pradius = float(conf[2])
                rad_neigh = float(conf[3])
                nn = int(conf[4])

                if len(conf) > 5:
                    if conf[5].isdigit():
                        device = conf[5]
                    else:
                        edge_attr = [attr.strip() for attr in conf[5].split('-')]
                        if len(conf) == 7:
                            device = conf[6]
                        elif len(conf) > 7:
                            raise RuntimeError("Invalid parameters in {} pool layer".format(d))
                module = ext.NVGraphPooling(pradius, aggr=aggr,
                                            neighs=nn, rad_neigh=rad_neigh,
                                            self_loop=True, edge_attr=edge_attr,
                                            flow=self.flow)

                nEdgeAttr = ext.numberEdgeAttr(edge_attr, nfeat)

            elif conf[0] == 'gp':
                if len(conf) < 2:
                    raise RuntimeError("Global Pooling Layer needs more arguments")
                aggr = conf[1]
                module = ext.GlobalPCPooling(aggr)

                if len(conf) == 3:
                    device = conf[2]
            # change edge atribs
            elif conf[0] == 'eg':
                if len(conf) > 1:
                    if conf[1].isdigit():
                        device = conf[1]

                    else:
                        edge_attr = [attr.strip() for attr in conf[1].split('-')]
                        if len(conf) == 3:
                            device = conf[2]
                        elif len(conf) > 3:
                            raise RuntimeError("Invalid parameters in {} edge_generation layer".format(d))

                module = ext.GraphReg(knn=None, edge_attr=edge_attr, flow=self.flow)
                nEdgeAttr = ext.numberEdgeAttr(edge_attr, nfeat)

            else:
                raise RuntimeError("{} layer does not exist".format(conf[0]))

            # Adding layer to modules
            if self.multigpu is True:
                if device is None:
                    raise RuntimeError("Multigpu is enabled and layer {} does not have a gpu assigned.".format(d))
                device = gpuid2device(device)
                self.devices.append(device)
                module = module.to(device)
            self.add_module(str(d), module)

    def obtain_intermediate(self, layer):

        self.intermediate = layer

    def forward(self, data):
        for i, module in enumerate(self._modules.values()):
            if self.multigpu:
                data = data.to(self.devices[i])
            if type(module) == torch.nn.Linear or \
               type(module) == torch.nn.BatchNorm1d or \
               type(module) == torch.nn.Dropout or \
               type(module) == torch.nn.ReLU:
                if (type(data) == torch_geometric.data.batch.Batch or
                        type(data) == torch_geometric.data.data.Data):

                    data.x = module(data.x)

                elif (type(data) == torch.Tensor):

                    data = module(data)

                else:
                    raise RuntimeError("Unknonw data type in forward time in {} module".format(type(module)))

            elif type(module) == ext.AGC:
                data.x = module(data.x, data.edge_index, data.edge_attr.float())

            elif type(module) == ext.NVGraphPooling or\
                    type(module) == ext.VGraphPooling or\
                    type(module) == ext.VPooling or\
                    type(module) == ext.NVPooling or\
                    type(module) == ext.MUNEGC or\
                    type(module) == ext.GraphReg:

                data = module(data)

            elif type(module) == ext.GlobalPCPooling:
                data = module(data.x, data.batch)

            else:
                raise RuntimeError("Unknown Module in forward time")

            if self.intermediate is not None:
                if self.intermediate == i:
                    return data

        return data
