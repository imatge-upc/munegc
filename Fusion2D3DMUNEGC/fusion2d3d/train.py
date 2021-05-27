"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import torch

import numpy as np
import os
import sys
import math
import argparse
from tqdm import tqdm

import ast

import twostream_network as models
from torch.utils.tensorboard import SummaryWriter
from pc_img_h5_dataset import PCIMGH5Dataset, custom_collate

import gc

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
        batch_1, batch_2 = batch
        if cuda:
            batch_1 = batch_1.to('cuda:0')
            batch_2 = batch_2.to('cuda:0')

        outputs = model(batch_1, batch_2)

        out_device = outputs.device
        gt = batch_1.y.to(out_device)
        loss = loss_criterion(outputs, gt)
        loss.backward()

        batch_size = len(torch.unique(batch_1.batch))

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
        batch_1, batch_2 = batch
        if cuda:
            batch_1 = batch_1.to('cuda:0')
            batch_2 = batch_2.to('cuda:0')

        batch_size = len(batch_1.batch.unique())

        outputs = model(batch_1, batch_2)
        out_device = outputs.device
        gt = batch_1.y.to(out_device)

        loss = loss_criterion(outputs, gt)

        loss_value = loss.detach().cpu().item()
        gt_value = gt.detach().cpu().data.numpy()
        outputs_value = outputs.detach().cpu().data.numpy()

        losses.update(loss_value, batch_size)

        cm.add(gt_value, outputs_value)

        loader.set_postfix({"loss": loss_value})
        torch.cuda.empty_cache()

    return losses.avg, cm


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='FusionClassificationStage')

    parser.add_argument('--optim', type=str, default='adam',
                        choices=['sgd', 'adam', 'radam'], help='str, optimizer. valid options sgd|adam|radam. Default adam')
    parser.add_argument('--wd', type=float, default=1e-4,  help='float, weight decay. Default 1e-4')
    parser.add_argument('--lr', type=float, default=1e-3,  help='float, learning rate. Default 1e-3')
    parser.add_argument('--momentum', type=float, default=0.9, help='float, momentum. Default 0.9')
    parser.add_argument('--betas', type=str, default='(0.9,0.999)', help="str, adam's betas. Default (0.9, 0.999)")
    parser.add_argument('--epochs', type=int, default=10,
                        help='int, number of epochs to train. If <=0, only testing will be done. Default 10')
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
    parser.add_argument('--train_split', type=str, default='list/train_list.txt',
                        help="str, path to the txt file that contains the list of files used on train")
    parser.add_argument('--val_split', type=str, default='list/test_list.txt',
                        help="str, path to the txt file that contains the list of files used on validation")
    parser.add_argument('--classname', type=str, default='list/scenes_labels.txt',
                        help='str, path to the file that contains the name of the classes')
    parser.add_argument('--dataset', type=str, default='nyu_v2', help='str, name of the dataset used')
    parser.add_argument('--weights', type=str, default='',
                        help='str, path to the file that contains the weighs assigned to each class.')
    parser.add_argument('--odir', type=str, default='./results/fusion',
                        help='str, path to the folder to be used to store the results')
    parser.add_argument('--exp_name', type=str, default='fusion', help='str, name of the current experiment')

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
    parser.add_argument('--seed', default=1, type=int, help='int, seed for random initialization. Default 1)')

    # Point cloud processing
    parser.add_argument('--pc_augm_input_dropout', type=float, default=0,
                        help='float, probability of removing points in input point clouds')
    parser.add_argument('--pc_augm_rot', default=False, action='store_true', help='bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', type=float, default=0.0,
                        help='float, probability of mirroring about x or y axis')

    args = parser.parse_args()
    args.betas = ast.literal_eval(args.betas)

    utils.seed(args.seed)

    exp_path = os.path.join(args.odir, args.dataset, (args.dataset_folder_b1 + '_' +
                                                      args.dataset_folder_b2).replace('/', '_'), args.exp_name.replace(" ", "_"))
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

    with open(os.path.join(exp_path, 'cmdline.txt'), 'w') as f:
        f.write(" ".join(sys.argv))

    train_writer = SummaryWriter(log_train_path)
    val_writer = SummaryWriter(log_val_path)

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

    if args.cuda is True and args.multigpu is False:
        model.to('cuda:0')

    print(model)
    parameters = model.parameters()
    if args.optim == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=args.betas, weight_decay=args.wd)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    elif args.optim == 'radam':
        optimizer = RAdam(parameters, lr=args.lr, betas=args.betas, weight_decay=args.wd)

    weights = None
    if args.weights != '' and args.weights != '-':
        weights_path = os.path.join(args.dataset_path, args.weights)
        if not os.path.isfile(weights_path):
            raise RuntimeError("weights file does not exist")
        weights = torch.FloatTensor([float(i) for i in utils.read_string_list(weights_path)]).cuda(args.lastgpu)

    loss_criterion = torch.nn.CrossEntropyLoss(weight=weights)

    label_path = os.path.join(args.dataset_path, args.classname)
    if not os.path.isfile(label_path):
        raise RuntimeError("label file does not exist")
    label_names = utils.read_string_list(label_path)

    transform3d = {"dropout": args.pc_augm_input_dropout,
                   "rot": args.pc_augm_rot,
                   "mirror": args.pc_augm_mirror_prob}

    train_dataset = PCIMGH5Dataset(args.dataset_path, args.dataset_folder_b1,
                                   args.dataset_folder_b2,
                                   args.train_split, transform3d=transform3d,
                                   pos_int16=args.pos_int16)

    val_dataset = PCIMGH5Dataset(args.dataset_path, args.dataset_folder_b1,
                                 args.dataset_folder_b2,
                                 args.val_split,
                                 pos_int16=args.pos_int16)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.nworkers,
                                               shuffle=True,
                                               drop_last=True,
                                               pin_memory=True,
                                               collate_fn=custom_collate
                                               )
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.nworkers,
                                             shuffle=False,
                                             pin_memory=True,
                                             collate_fn=custom_collate
                                             )

    is_best_meanacc = False
    best_meanacc = 0
    start_epoch = 0

    if start_epoch == (args.epochs-1):
        print('Training already finished, stopping job')
        exit()

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

        print('-> Train: \tLoss: {}, \tMeanAcc: {}'.format(train_loss, train_meanacc))

        val_loss, val_cm = val(model, val_loader,
                               loss_criterion,
                               label_names,
                               cuda=args.cuda)

        val_meanacc = val_cm.mean_acc()
        val_writer.add_scalar('Loss', val_loss, epoch)
        val_writer.add_figure('Confusion_Matrix', val_cm.plot(normalize=True), epoch)
        val_writer.add_scalar('Mean_Acc', val_meanacc, epoch)

        torch.cuda.empty_cache()

        print('-> Val: \tLoss: {}, \tMean Acc: {}'.format(val_loss, val_meanacc))
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
