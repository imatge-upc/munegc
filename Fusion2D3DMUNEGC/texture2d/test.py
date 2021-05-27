"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import torch
import torchvision.transforms as transforms
import torchvision.models as models

from tqdm import tqdm
import argparse
import os

from img_h5_dataset import ImageH5Dataset

from Fusion2D3DMUNEGC.utilities import utils
from Fusion2D3DMUNEGC.utilities import metrics


@torch.no_grad()
def test(model, loader, label_names, cuda=True):
    model.eval()
    loader = tqdm(loader, ncols=100)
    cm = metrics.ConfusionMatrixMeter(label_names, cmap='Blues')
    for i, batch in enumerate(loader, start=0):
        img, gt = batch
        if cuda:
            img = img.to('cuda:0')
            gt = gt.to('cuda:0')

        pred = model(img)

        cm.add(gt.cpu().data.numpy(), pred.cpu().data.numpy())
    return cm.mean_acc()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='2DTextureBranch')

    parser.add_argument('--batch_size', type=int, default=16, help='int, batch size. Default 16')

    parser.add_argument('--cuda', default=False, action='store_true', help='Bool, activate cuda')
    parser.add_argument('--nworkers', type=int, default=4,
                        help='int, num subprocesses to use for data loading. 0 means that the data will be loaded in the main process')

    parser.add_argument('--dataset_path', type=str, default='datasets/nyu_v2',
                        help='str, root path to the dataset folder')
    parser.add_argument('--dataset_folder', type=str, default='h5/h5_feat2d',
                        help='str, folder that contains the h5 files')
    parser.add_argument('--test_split', type=str, default='list/test_list.txt',
                        help='str, path to the txt file that contains the list of files used on testing')
    parser.add_argument('--classname', type=str, default='list/scenes_labels.txt',
                        help='str, path to the file that contains the name of the classes')
    parser.add_argument('--nclass', type=int, default='10', help='int, number of classes of this dataset')
    parser.add_argument('--range01', default=False, action='store_true', help='bool, normalize features to 0-1 range')

    parser.add_argument('--pretrain_path', type=str, default='-', help='str, path to the checkpoint to be used')

    args = parser.parse_args()

    model = models.resnet18(num_classes=int(args.nclass))

    print('loading pretrain')

    if (os.path.exists(args.pretrain_path)):
        _, _, model_state, _ = utils.load_checkpoint(args.pretrain_path)
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

    test_dataset = ImageH5Dataset(args.dataset_path, args.dataset_folder,
                                  args.test_split, transform=normalize,
                                  range01=bool(args.range01))

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=int(args.batch_size),
                                              num_workers=args.nworkers,
                                              shuffle=False,
                                              pin_memory=True)

    mean_acc = test(model, test_loader,
                    label_names,
                    cuda=args.cuda)

    print("Mean acc: ", mean_acc)
