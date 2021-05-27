"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import h5py
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
import os
import torch
import resnet_mod as resnet
from img_h5_dataset_fname import ImageH5Dataset

from Fusion2D3DMUNEGC.utilities import utils
from Fusion2D3DMUNEGC.utilities import metrics


def save_h5_features(fname_h5_feat2d, feat):
    h5_feat = h5py.File(fname_h5_feat2d.strip(), "a")

    h5_feat.create_dataset(
        "feat", data=feat,
        compression='gzip', compression_opts=4,
        dtype='float')

    h5_feat.close()


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
        img, gt, files = batch
        if cuda:
            img = img.to('cuda:0')
            gt = gt.to('cuda:0')

        pred = model(img)

        cm.add(gt.cpu().data.numpy(), pred.cpu().data.numpy())
    return cm.mean_acc()


@torch.no_grad()
def create_feat(model, loader, path_h5_feat2d, nLayer, cuda=True):

    model.eval()
    loader = tqdm(loader, ncols=100)

    print("This layer will be used to extract the features: ", list(model.children())[nLayer][1])

    activations = SaveFeatures(list(model.children())[nLayer][1].sum)

    for i, batch in enumerate(loader, start=0):
        img, gt, files = batch
        if cuda:
            img = img.to('cuda:0')
            gt = gt.to('cuda:0')

        _ = model(img)
        features = activations.features
        for i in range(0, features.size(0)):
            feature = features[i, :, :, :]
            feature = feature.view(features.size(1), -1).permute(1, 0)

            fname_h5_feat2d = os.path.join(path_h5_feat2d, files[i]+".h5")
            save_h5_features(fname_h5_feat2d, feature.detach().cpu().numpy())

    activations.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='2DTextureFeatExtract')

    parser.add_argument('--batch_size', type=int, default=16, help='int, batch size. Default 16')

    parser.add_argument('--cuda', default=False, action='store_true', help='Bool, activate cuda')
    parser.add_argument('--nworkers', type=int, default=4,
                        help='int, num subprocesses to use for data loading. 0 means that the data will be loaded in the main process')

    parser.add_argument('--dataset_path', type=str, default='datasets/nyu_v2',
                        help='str, root path to the dataset folder')
    parser.add_argument('--dataset_folder', type=str, default='h5/h5_2dimg',
                        help='str, folder that contains the h5 files')

    parser.add_argument('--test_split', type=str, default='list/test_list.txt',
                        help='str, path to the txt file that contains the list of files used on validation')
    parser.add_argument('--dataset', type=str, default='list/dataset.txt',
                        help='str, path to the txt file that contains the list of files of the whole dataset')
    parser.add_argument('--classname', type=str, default='list/scenes_labels.txt',
                        help='str, path to the file that contains the name of the classes')
    parser.add_argument('--nclass', type=int, default='10', help='int, number of classes of this dataset')
    parser.add_argument('--range01', default=False, action='store_true', help='bool, normalize features to 0-1 range')

    parser.add_argument('--pretrain_path', type=str, default='-', help='str, path to the checkpoint to be used')

    parser.add_argument('--nlayer', default=1, type=int, help='int, layer to be used to extract the features')
    parser.add_argument('--check_accuracy', default=False, action='store_true',
                        help='bool, check mean accuracy obtained')
    parser.add_argument('--h5_feat_2d', type=str, default='h5/h5_feat_2d', help='str, folder to store the 2d feat')

    args = parser.parse_args()

    model = resnet.resnet18(num_classes=args.nclass)

    print('loading pretrain')

    if (os.path.exists(args.pretrain_path)):
        _, stored_mean_acc, model_state, _ = utils.load_checkpoint(args.pretrain_path)
        model.load_state_dict(model_state)

    else:
        print('Wrong pretrain path')
        exit()

    print(model)

    if args.cuda is True:
        model = model.to('cuda:0')

    label_path = os.path.join(args.dataset_path, args.classname)
    if not os.path.isfile(label_path):
        raise RuntimeError("label file does not exist")
    label_names = utils.read_string_list(label_path)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([normalize, ])

    dataset = ImageH5Dataset(args.dataset_path, args.dataset_folder,
                             args.dataset, transform=transform,
                             range01=args.range01)

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=int(args.batch_size),
                                         num_workers=args.nworkers,
                                         shuffle=False,
                                         pin_memory=True
                                         )
    if args.check_accuracy is True:
        test_dataset = ImageH5Dataset(args.dataset_path, args.dataset_folder,
                                      args.test_split, transform=transform,
                                      range01=args.range01)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=int(args.batch_size),
                                                  num_workers=args.nworkers,
                                                  shuffle=False,
                                                  pin_memory=True
                                                  )

        mean_acc = check_accuracy(model, test_loader, label_names, cuda=args.cuda)

        print("Stored Mean acc: ", stored_mean_acc)
        print("Mean acc: ", mean_acc)

    path_h5_feat2d = os.path.join(args.dataset_path, args.h5_feat_2d)
    utils.create_folder(path_h5_feat2d)
    create_feat(model, loader, path_h5_feat2d, args.nlayer, cuda=args.cuda)
