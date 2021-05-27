"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np
from tqdm import tqdm
import argparse
import math
import os
import sys
import ast
import gc

from img_h5_dataset import ImageH5Dataset

from Fusion2D3DMUNEGC.utilities.radam import RAdam
from Fusion2D3DMUNEGC.utilities import utils
from Fusion2D3DMUNEGC.utilities import metrics


def train(model, loader, loss_criteron, optimizer, label_names, cuda=True):
    model.train()
    loader = tqdm(loader, ncols=100)
    losses = metrics.AverageMeter()
    cm = metrics.ConfusionMatrixMeter(label_names, cmap='Oranges')
    optimizer.zero_grad()
    for i, batch in enumerate(loader, start=0):
        img, gt = batch
        if cuda:
            img = img.to('cuda:0')
            gt = gt.to('cuda:0')

        outputs = model(img)

        out_device = outputs.device
        gt = gt.to(out_device)
        loss = loss_criterion(outputs, gt)
        loss.backward()

        batch_size = img.size(0)

        loss_value = loss.detach().cpu().item()
        gt_value = gt.detach().cpu().data.numpy()
        outputs_value = outputs.detach().cpu().data.numpy()

        losses.update(loss_value, batch_size)
        cm.add(gt_value, outputs_value)

        loader.set_postfix({"loss": loss_value})

        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.empty_cache()
    return losses.avg, cm


@torch.no_grad()
def val(model, loader, loss_criterion, label_names, cuda=True):
    model.eval()
    loader = tqdm(loader, ncols=100)
    losses = metrics.AverageMeter()
    cm = metrics.ConfusionMatrixMeter(label_names, cmap='Blues')

    for i, batch in enumerate(loader, start=0):
        img, gt = batch
        if cuda:
            img = img.to('cuda:0')
            gt = gt.to('cuda:0')

        outputs = model(img)

        out_device = outputs.device
        gt = gt.to(out_device)

        loss = loss_criterion(outputs, gt)

        batch_size = img.size(0)

        loss_value = loss.detach().cpu().item()
        gt_value = gt.detach().cpu().data.numpy()
        outputs_value = outputs.detach().cpu().data.numpy()

        losses.update(loss_value, batch_size)

        cm.add(gt_value, outputs_value)

        loader.set_postfix({"loss": loss_value})
        torch.cuda.empty_cache()

    return losses.avg, cm


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='2DTextureBranch')

    parser.add_argument('--optim', type=str, default='adam',
                        choices=['sgd', 'adam', 'radam'], help='str, optimizer. valid options sgd|adam|radam. Default adam')
    parser.add_argument('--wd', type=float, default=1e-4,  help='float, weight decay. Default 1e-4')
    parser.add_argument('--lr', type=float, default=1e-3,  help='float, learning rate. Default 1e-3')
    parser.add_argument('--momentum', type=float, default=0.9, help='float, momentum. Default 0.9')
    parser.add_argument('--betas', type=str, default='(0.9,0.999)', help="str, adam's betas. Default (0.9, 0.999)")
    parser.add_argument('--epochs', type=int, default=10,
                        help='int, number of epochs to train. If <=0, only testing will be done. Default 10')
    parser.add_argument('--batch_size', type=int, default=16, help='int, batch size. Default 16')

    parser.add_argument('--cuda', default=False, action='store_true', help='Bool, activate cuda')
    parser.add_argument('--nworkers', type=int, default=4,
                        help='int, num subprocesses to use for data loading. 0 means that the data will be loaded in the main process')

    parser.add_argument('--dataset_path', type=str, default='datasets/nyu_v2',
                        help='str, root path to the dataset folder')
    parser.add_argument('--dataset_folder', type=str, default='h5/h5_feat2d',
                        help='str, folder that contains the h5 files')
    parser.add_argument('--train_split', default='list/train_list.txt',
                        help='str, path to the txt file that contains the list of files used on train')
    parser.add_argument('--val_split', type=str, default='list/test_list.txt',
                        help='str, path to the txt file that contains the list of files used on validation')
    parser.add_argument('--classname', type=str, default='list/scenes_labels.txt',
                        help='str, path to the file that contains the name of the classes')
    parser.add_argument('--nclass', type=int, default='10', help='int, number of classes of this dataset')
    parser.add_argument('--dataset', type=str, default='nyu_v2', help='str, name of the dataset used')
    parser.add_argument('--weights', type=str, default='',
                        help='str, path to the file that contains the weighs assigned to each class')
    parser.add_argument('--odir', type=str, default='./results/texture2d',
                        help='str, path to the folder to be used to store the results')
    parser.add_argument('--exp_name', type=str, default='geometric3dbranch', help='str, name of the current experiment')
    parser.add_argument('--range01', default=False, action='store_true', help='bool, normalize features to 0-1 range')

    parser.add_argument('--transfer_learning', default=False, action='store_true',
                        help='Bool, activate transfer learning')
    parser.add_argument('--places', default=False, action='store_true', help='bool, transfer learning from places')
    parser.add_argument('--tl_path', type=str, default='',
                        help='str, path to the weighs used to do the transfer learning')
    parser.add_argument('--tl_nclass', type=int, default=19, help='Pretrain number of classes')

    parser.add_argument('--seed', default=1, type=int, help='int, seed for random initialization. Default 1)')

    args = parser.parse_args()
    args.betas = ast.literal_eval(args.betas)

    utils.seed(args.seed)

    exp_path = os.path.join(args.odir, args.dataset, args.dataset_folder, args.exp_name.replace(" ", "_"))

    print("The experiment will be saved in: " + exp_path)
    utils.create_folder(args.odir)
    utils.create_folder(exp_path)
    log_path = os.path.join(exp_path, 'log')
    utils.create_folder(log_path)
    log_train_path = os.path.join(log_path, 'train')
    utils.create_folder(log_train_path)
    log_val_path = os.path.join(log_path, 'val')
    utils.create_folder(log_val_path)
    checkpoint_path = os.path.join(exp_path, 'checkpoints')
    utils.create_folder(checkpoint_path)
    cm_path = os.path.join(exp_path, 'cm')
    utils.create_folder(cm_path)

    with open(os.path.join(exp_path, 'cmdline.txt'), 'w') as f:
        f.write(" ".join(sys.argv))

    train_writer = SummaryWriter(log_train_path)
    val_writer = SummaryWriter(log_val_path)

    if args.transfer_learning is True:
        print('loading pretrain model')
        model = models.resnet18(num_classes=args.tl_nclass)
        checkpoint = torch.load(args.tl_path, map_location=lambda storage, loc: storage)
        if bool(args.places) is True:
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        else:
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['model'].items()}
        model.load_state_dict(state_dict)
        in_channels = model.fc.in_features
        model.fc = torch.nn.Linear(in_channels, args.nclass)
        torch.nn.init.constant_(model.fc.bias, -np.log(args.nclass-1))

    else:
        model = models.resnet18(num_classes=args.nclass)

    if args.cuda is True:
        model.to('cuda:0')

    print(model)

    parameters = model.parameters()
    if args.optim == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=args.betas, weight_decay=args.wd)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.optim == 'radam':
        print('radam')
        optimizer = RAdam(parameters, lr=args.lr, betas=args.betas, weight_decay=args.wd)

    weights = None
    if args.weights != '' and args.weights != '-':
        weights_path = os.path.join(args.dataset_path, args.weights)
        if not os.path.isfile(weights_path):
            raise RuntimeError("weights file does not exist")
        weights = torch.FloatTensor([float(i) for i in utils.read_string_list(weights_path)]).cuda(0)

    loss_criterion = torch.nn.CrossEntropyLoss(weight=weights)

    label_path = os.path.join(args.dataset_path, args.classname)
    if not os.path.isfile(label_path):
        raise RuntimeError("label file does not exist")
    label_names = utils.read_string_list(label_path)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        normalize,
    ])

    train_dataset = ImageH5Dataset(args.dataset_path, args.dataset_folder,
                                   args.train_split, transform=train_transform,
                                   range01=args.range01)

    val_dataset = ImageH5Dataset(args.dataset_path, args.dataset_folder,
                                 args.val_split, transform=val_transform,
                                 range01=args.range01)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               args.batch_size,
                                               num_workers=args.nworkers,
                                               shuffle=True,
                                               drop_last=True,
                                               pin_memory=False
                                               )
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.nworkers,
                                             shuffle=False,
                                             pin_memory=False
                                             )

    start_epoch = 0
    is_best_meanacc = False
    best_meanacc = 0

    for epoch in range(start_epoch, args.epochs):

        print('Epoch {}/{} ({}):'.format(epoch, args.epochs, args.exp_name))

        train_loss, train_cm = train(model, train_loader,
                                     loss_criterion,
                                     optimizer,
                                     label_names,
                                     cuda=args.cuda)

        train_meanacc = train_cm.mean_acc()

        train_writer.add_scalar('Loss', train_loss, epoch)
        train_writer.add_figure('Confusion_Matrix', train_cm.plot(normalize=True), epoch)
        train_writer.add_scalar('Mean_Acc', train_meanacc, epoch)

        torch.cuda.empty_cache()

        print('-> Train:\tLoss: {}, \tMeanAcc: {}'.format(train_loss, train_meanacc))

        val_loss, val_cm = val(model, val_loader,
                               loss_criterion,
                               label_names,
                               cuda=args.cuda)

        val_meanacc = val_cm.mean_acc()

        val_writer.add_scalar('Loss', val_loss, epoch)
        val_writer.add_figure('Confusion_Matrix', val_cm.plot(normalize=True), epoch)
        val_writer.add_scalar('Mean_Acc', val_meanacc, epoch)

        torch.cuda.empty_cache()

        print('-> Val:\tLoss: {}, \tMean Acc: {}'.format(val_loss, val_meanacc))
        is_best_meanacc = val_meanacc > best_meanacc

        if is_best_meanacc:
            best_meanacc = val_meanacc
            val_writer.add_text('Best Mean Accuracy', str(np.round(best_meanacc.item(), 2)), epoch)

        utils.save_checkpoint(epoch, model, optimizer, val_meanacc, best_meanacc,
                              is_best_meanacc, checkpoint_path, save_all=False)

        if math.isnan(val_loss) or math.isnan(train_loss):
            break

        train_writer.flush()
        val_writer.flush()
        del val_loss, val_cm, train_loss, train_cm
        gc.collect()
    train_writer.close()
    val_writer.close()
