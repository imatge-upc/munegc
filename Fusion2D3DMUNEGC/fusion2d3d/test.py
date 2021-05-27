"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import torch

import os
import argparse
from tqdm import tqdm

import twostream_network as models
from pc_img_h5_dataset import PCIMGH5Dataset, custom_collate

from Fusion2D3DMUNEGC.utilities import utils
from Fusion2D3DMUNEGC.utilities import metrics


@torch.no_grad()
def test(model, loader, label_names, cuda=True):
    model.eval()
    loader = tqdm(loader, ncols=100)
    cm = metrics.ConfusionMatrixMeter(label_names, cmap='Blues')

    for i, batch in enumerate(loader, start=0):
        batch_1, batch_2 = batch
        if cuda:
            batch_1 = batch_1.to('cuda:0')
            batch_2 = batch_2.to('cuda:0')

        outputs = model(batch_1, batch_2)
        out_device = outputs.device
        gt = batch_1.y.to(out_device)

        gt_value = gt.detach().cpu().data.numpy()
        outputs_value = outputs.detach().cpu().data.numpy()

        cm.add(gt_value, outputs_value)

        torch.cuda.empty_cache()
    return cm.mean_acc()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='FusionClassificationStage')

    parser.add_argument('--batch_size', type=int, default=32, help='int, batch size. Default 32')
    parser.add_argument('--cuda', default=False, action='store_true', help='Bool, activate cuda')
    parser.add_argument('--multigpu', default=False, action='store_true', help='Bool, activate multigpu')
    parser.add_argument('--lastgpu', type=int, default=0,
                        help='int, parameter to indicate which is the last gpu in multigpu scenarios')

    parser.add_argument('--nworkers', type=int, default=4,
                        help='int, num subprocesses to use for data loading. 0 means that the data will be loaded in the main process')

    parser.add_argument('--dataset_path', type=str, default='datasets/nyu_v2',
                        help='str, root path to the dataset folder')
    parser.add_argument('--dataset_folder_b1', type=str, default='h5/h5_feat3d',
                        help="str, folder that contains the h5 files for branch 1")
    parser.add_argument('--dataset_folder_b2', type=str, default='h5/h5_feat2d',
                        help='str, folder that contains the h5 files for branch 2')
    parser.add_argument('--test_split', type=str, default='list/test_list.txt',
                        help="str, path to the txt file that contains the list of files used on testing")
    parser.add_argument('--classname', type=str, default='list/scenes_labels.txt',
                        help='str, path to the file that contains the name of the classes')
    parser.add_argument('--dataset', type=str, default='nyu_v2', help='str, name of the dataset used')

    parser.add_argument('--pos_int16', default=False, action='store_true',
                        help='bool, positions are encoded with int16')
    parser.add_argument('--nfeatures_b1', type=int, default='128',
                        help='int, number of features used as input on branch 1')
    parser.add_argument('--nfeatures_b2', type=int, default='512',
                        help='int, number of features used as input on branch 2')
    parser.add_argument('--proj_b1', default=False, action='store_true',
                        help='bool, activate projection function in branch 1')
    parser.add_argument('--proj_b2', default=False, action='store_true',
                        help='bool, activate projection function in branch 2')
    parser.add_argument('--features_proj_b1', type=int, default='256',
                        help='number of output channels of the projection in branch 1')
    parser.add_argument('--features_proj_b2', type=int, default='256',
                        help='number of output channels of the projection in branch 2')
    parser.add_argument('--rad_fuse_pool', type=float, default=0.35,
                        help='float, radius used to create the voxel used to fuse both branches')

    parser.add_argument('--classification_model', type=str, default='gp_avg, b, r, d_0.5, f_10_cp_1',
                        help='str, defines the model as a sequence of layers')

    parser.add_argument('--pretrain_path', type=str, default='-', help='str, path to the checkpoint to be used')

    args = parser.parse_args()

    features_b1 = args.nfeatures_b1
    features_b2 = args.nfeatures_b2
    features_proj_b1 = args.features_proj_b1
    features_proj_b2 = args.features_proj_b2

    model = models.TwoStreamNetwork(args.classification_model, features_b1,
                                    features_b2, args.rad_fuse_pool,
                                    args.multigpu,
                                    features_proj_b1=features_proj_b1,
                                    features_proj_b2=features_proj_b2,
                                    proj_b1=args.proj_b1,
                                    proj_b2=args.proj_b2)

    print('loading pretrain')

    if (os.path.isfile(args.pretrain_path)):
        _, _, model_state, _ = utils.load_checkpoint(args.pretrain_path)
        model.load_state_dict(model_state)
    else:
        print('Wrong pretrain path')
        exit()

    print(model)

    if args.cuda is True and args.multigpu is False:
        model.to('cuda:0')

    label_path = os.path.join(args.dataset_path, args.classname)
    if not os.path.isfile(label_path):
        raise RuntimeError("label file does not exist")
    label_names = utils.read_string_list(label_path)

    test_dataset = PCIMGH5Dataset(args.dataset_path, args.dataset_folder_b1,
                                  args.dataset_folder_b2,
                                  args.test_split,
                                  pos_int16=args.pos_int16)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.nworkers,
                                              shuffle=False,
                                              pin_memory=True,
                                              collate_fn=custom_collate
                                              )
    mean_acc = test(model, test_loader,
                    label_names,
                    cuda=args.cuda)

    print("Mean acc: ", mean_acc)
