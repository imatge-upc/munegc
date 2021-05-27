"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import torch
import torch_geometric

import os
import argparse
from tqdm import tqdm

import ast

import graph_model as models
from pc_h5_dataset import PCH5Dataset

from Fusion2D3DMUNEGC.utilities import utils
from Fusion2D3DMUNEGC.utilities import metrics


@torch.no_grad()
def test(model, loader, label_names, cuda=True):
    model.eval()
    loader = tqdm(loader, ncols=100)
    cm = metrics.ConfusionMatrixMeter(label_names, cmap='Blues')

    for i, batch in enumerate(loader, start=0):
        if cuda:
            batch = batch.to('cuda:0')

        outputs = model(batch)
        out_device = outputs.device
        gt = batch.y.to(out_device)

        gt_value = gt.detach().cpu().data.numpy()
        outputs_value = outputs.detach().cpu().data.numpy()

        cm.add(gt_value, outputs_value)

        torch.cuda.empty_cache()

    return cm.mean_acc()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='3D Geometric branch')

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

    if (os.path.isfile(args.pretrain_path)):
        _, _, model_state, _ = utils.load_checkpoint(args.pretrain_path)
        model.load_state_dict(model_state)
    else:
        print('Wrong pretrain path')
        exit()

    print(model)

    if args.cuda is True and args.multigpu is False:
        model = model.to('cuda:0')

    label_path = os.path.join(args.dataset_path, args.classname)
    if not os.path.isfile(label_path):
        raise RuntimeError("label file does not exist")
    label_names = utils.read_string_list(label_path)
    assert args.batch_size % args.batch_parts == 0

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
    mean_acc = test(model, test_loader,
                    label_names,
                    cuda=args.cuda)

    print("Mean acc: ", mean_acc)
