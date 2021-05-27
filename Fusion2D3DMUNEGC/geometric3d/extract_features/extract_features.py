"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import ast
import h5py
from tqdm import tqdm
import argparse
import os
import torch
import torch_geometric

from pc_h5_dataset_findx import PCH5Dataset
import Fusion2D3DMUNEGC.geometric3d.graph_model as models
from Fusion2D3DMUNEGC.utilities import utils
from Fusion2D3DMUNEGC.utilities import metrics


def save_h5_features(filename_feat, feat, pos, label):
    h5 = h5py.File(filename_feat, "w")

    h5.create_dataset(
        "points", data=pos,
        compression='gzip', compression_opts=4,
        dtype='float')

    h5.create_dataset(
        "feat", data=feat,
        compression='gzip', compression_opts=4,
        dtype='float')

    h5.create_dataset(
        "label", data=label,
        dtype='uint8')

    h5.close()


class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()


@torch.no_grad()
def check_accuracy(model, loader, label_names, cuda=True):
    model.eval()
    loader = tqdm(loader, ncols=100)
    cm = metrics.ConfusionMatrixMeter(label_names, cmap='Blues')

    for i, batch in enumerate(loader, start=0):
        if cuda:
            batch = batch.to('cuda:0')
        outputs = model(batch)
        out_device = outputs.device
        gt = batch.y.to(out_device)

        cm.add(gt.cpu().data.numpy(), outputs.cpu().data.numpy())

    return cm.mean_acc()


@torch.no_grad()
def extract_features(model, loader, split_path, h5_path, nLayer, cuda=True):
    model.eval()

    model.obtain_intermediate(nLayer)

    features = SaveFeatures(list(model.children())[nLayer])

    print("This layer will be used to extract the features: ", list(model.children())[nLayer])
    loader = tqdm(loader, ncols=100)

    files = utils.read_string_list(split_path)
    for i, batch in enumerate(loader, start=0):

        if cuda:
            batch = batch.to('cuda:0')

        graph = model(batch)

        feat = features.features
        if type(feat) == torch_geometric.data.batch.Batch:
            feat = feat.x
        if type(graph) == torch_geometric.data.batch.Batch:
            if feat.size(0) == graph.x.size(0) and graph.x.size(0) == graph.pos.size(0) and graph.pos.size(0) == len(graph.batch):
                graph.x = feat
                x, _ = torch_geometric.utils.to_dense_batch(graph.x, batch=graph.batch)
                pos, _ = torch_geometric.utils.to_dense_batch(graph.pos, batch=graph.batch)
                labels = graph.y
                file_indexes = graph.c
                for i in range(0, x.size(0)):
                    x_i = x[i, :, :]
                    pos_i = pos[i, :, :]
                    y_i = labels[i]
                    h5_file_name = os.path.join(h5_path, files[file_indexes[i]]+'.h5')
                    save_h5_features(h5_file_name,
                                     x_i.detach().cpu().numpy(),
                                     pos_i.detach().cpu().numpy(),
                                     y_i.detach().cpu().numpy())
            else:
                print('Dimensions doesn\'t match')
                exit()
        else:
            raise RuntimeError('wrong input data')
    features.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='3D Geometric branch feat extract')

    parser.add_argument('--batch_size', type=int, default=32, help='int, batch size. Default 32')
    parser.add_argument('--batch_parts', type=int, default=1,
                        help='int, batch can be evaluated sequentially in multiple shards, should be >=1, very useful in low memory settings, though computation is not strictly equivalent due to batch normalization runnning statistics. Default 1')
    parser.add_argument('--cuda', default=False, action='store_true', help='Bool, activate cuda')
    parser.add_argument('--multigpu', default=False, action='store_true', help='Bool, activate multigpu')
    parser.add_argument('--lastgpu', type=int, default=0,
                        help='int, parameter to indicate which is the last gpu in multigpu scenarios')
    parser.add_argument('--nworkers', type=int, default=4,
                        help='int, num subprocesses to use for data loading. 0 means that the data will be loaded in the main process')

    parser.add_argument('--dataset_path', type=str, default='datasets/nyu_v2',
                        help='str, root path to the dataset folder')
    parser.add_argument('--dataset_folder', type=str, default='h5/h5_3dhha',
                        help='str, folder that contains the h5 files')
    parser.add_argument('--dataset', type=str, default='list/dataset.txt',
                        help='str, path to the txt file that contains the list of files of the whole dataset')
    parser.add_argument('--test_split', type=str, default='list/test_list.txt',
                        help='str, path to the txt file that contains the list of files used on testing')
    parser.add_argument('--classname', type=str, default='list/scenes_labels.txt',
                        help='str, path to the file that contains the name of the classes')
    parser.add_argument('--nfeatures', type=int, default='3', help='int, number of features used as input')
    parser.add_argument('--range01', default=False, action='store_true', help='bool, normalize features to 0-1 range')
    parser.add_argument('--pos_int16', default=False, action='store_true',
                        help='bool, positions are encoded with int16')

    parser.add_argument('--model_config', type=str, default='-',
                        help='str, defines the model as a sequence of layers')

    parser.add_argument('--fnet_widths', type=str,
                        default='[128]', help='str, list of widths of the hidden filter net layers (excluding the input and output ones, they are automatically calculated)')
    parser.add_argument('--fnet_llbias', default=False, action='store_true',
                        help='bool, use bias in the last layer in filter gen net')
    parser.add_argument('--fnet_tanh', default=False, action='store_true',
                        help='bool, use tanh as output activation in filter gen net.')
    parser.add_argument('--conv_bias', default=False, action='store_true', help='bool, use bias for convolutions')
    parser.add_argument('--edge_attr', type=str, default='posspherical-featureoffsets',
                        help='str, edge attributes. Default posspherical-featureoffsets')

    parser.add_argument('--pretrain_path', type=str, default='-', help='str, path to the checkpoint to be used')

    parser.add_argument('--nlayer', default=1, type=int, help='int, layer to be used to extract the features')
    parser.add_argument('--check_accuracy', default=False, action='store_true',
                        help='bool, check mean accuracy obtained')
    parser.add_argument('--h5_feat_3d', type=str, default='h5/h5_feat_3d', help='str, folder to store the 2d feat')

    args = parser.parse_args()
    args.fnet_widths = ast.literal_eval(args.fnet_widths)

    features = args.nfeatures

    model = models.GraphNetwork(args.model_config, features, multigpu=args.multigpu,
                                default_fnet_widths=args.fnet_widths,
                                default_fnet_llbias=args.fnet_llbias,
                                default_edge_attr=args.edge_attr,
                                default_conv_bias=args.conv_bias,
                                default_fnet_tanh=args.fnet_tanh)
    print('loading pretrain')

    if (os.path.exists(args.pretrain_path)):
        _, stored_mean_acc, model_state, _ = utils.load_checkpoint(args.pretrain_path)
        model.load_state_dict(model_state)

    else:
        print('Wrong pretrain path')
        exit()

    if args.cuda is True and args.multigpu is False:
        model = model.to('cuda:0')

    print(model)

    label_path = os.path.join(args.dataset_path, args.classname)
    if not os.path.isfile(label_path):
        raise RuntimeError("label file does not exist")
    label_names = utils.read_string_list(label_path)

    assert args.batch_size % args.batch_parts == 0

    dataset = PCH5Dataset(args.dataset_path, args.dataset_folder,
                          args.dataset,
                          range01=args.range01,
                          pos_int16=args.pos_int16)

    loader = torch_geometric.data.DataLoader(dataset,
                                             batch_size=int(args.batch_size/args.batch_parts),
                                             num_workers=args.nworkers,
                                             shuffle=False,
                                             pin_memory=False
                                             )

    if args.check_accuracy is True:

        test_dataset = PCH5Dataset(args.dataset_path, args.dataset_folder,
                                   args.test_split,
                                   range01=args.range01,
                                   pos_int16=args.pos_int16)

        test_loader = torch_geometric.data.DataLoader(test_dataset,
                                                      batch_size=int(args.batch_size/args.batch_parts),
                                                      num_workers=args.nworkers,
                                                      shuffle=False,
                                                      pin_memory=True
                                                      )

        mean_acc = check_accuracy(model, test_loader, label_names, cuda=args.cuda)

        print("Stored Mean acc: ", stored_mean_acc)
        print("Mean acc: ", mean_acc)

    split_path = os.path.join(args.dataset_path, args.dataset)
    h5_path = os.path.join(args.dataset_path, args.h5_feat_3d)
    utils.create_folder(h5_path)

    extract_features(model,
                     loader,
                     split_path,
                     h5_path,
                     args.nlayer,
                     cuda=args.cuda)
