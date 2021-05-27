"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import torch
import torch_geometric

import numpy as np
import os
import sys
import math
import argparse
from tqdm import tqdm

import ast

import graph_model as models
from torch.utils.tensorboard import SummaryWriter
from pc_h5_dataset import PCH5Dataset

import gc

from Fusion2D3DMUNEGC.utilities.radam import RAdam
from Fusion2D3DMUNEGC.utilities import utils
from Fusion2D3DMUNEGC.utilities import metrics


def train(model, loader, loss_criteron, optimizer, label_names, batch_parts=0, cuda=True):
    model.train()
    numIt = len(loader)
    loader = tqdm(loader, ncols=100)
    losses = metrics.AverageMeter()
    cm = metrics.ConfusionMatrixMeter(label_names, cmap='Oranges')
    prev = -1
    optimizer.zero_grad()
    for i, batch in enumerate(loader, start=0):
        if cuda:
            batch = batch.to('cuda:0')
        outputs = model(batch)

        out_device = outputs.device
        gt = batch.y.to(out_device)
        loss = loss_criterion(outputs, gt)

        loss.backward()

        batch_size = len(torch.unique(batch.batch))

        loss_value = loss.detach().cpu().item()
        gt_value = gt.detach().cpu().data.numpy()
        outputs_value = outputs.detach().cpu().data.numpy()

        losses.update(loss_value, batch_size)
        cm.add(gt_value, outputs_value)

        loader.set_postfix({"loss": loss_value})

        if (i+1) % batch_parts == 0 or (i+1) == numIt:
            if batch_parts > 1:
                accum = i-prev
                prev = i
                for p in model.parameters():
                    p.grad.div_(accum)
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
        if cuda:
            batch = batch.to('cuda:0')

        outputs = model(batch)
        out_device = outputs.device
        gt = batch.y.to(out_device)

        loss = loss_criterion(outputs, gt)

        batch_size = len(torch.unique(batch.batch))

        loss_value = loss.detach().cpu().item()
        gt_value = gt.detach().cpu().data.numpy()
        outputs_value = outputs.detach().cpu().data.numpy()

        losses.update(loss_value, batch_size)

        cm.add(gt_value, outputs_value)

        loader.set_postfix({"loss": loss_value})
        torch.cuda.empty_cache()

    return losses.avg, cm


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='3DGeometricBranch')

    parser.add_argument('--optim', type=str, default='adam',
                        choices=['sgd', 'adam', 'radam'], help='str, optimizer. valid options sgd|adam|radam. Default adam')
    parser.add_argument('--wd', type=float, default=1e-4,  help='float, weight decay. Default 1e-4')
    parser.add_argument('--lr', type=float, default=1e-3,  help='float, learning rate. Default 1e-3')
    parser.add_argument('--momentum', type=float, default=0.9, help='float, momentum. Default 0.9')
    parser.add_argument('--betas', type=str, default='(0.9,0.999)', help="str, adam's betas. Default (0.9, 0.999)")
    parser.add_argument('--epochs', type=int, default=10,
                        help='int, number of epochs to train. If <=0, only testing will be done. Default 10')
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
    parser.add_argument('--train_split', type=str, default='list/train_list.txt',
                        help='str, path to the txt file that contains the list of files used on train')
    parser.add_argument('--val_split', type=str, default='list/test_list.txt',
                        help='str, path to the txt file that contains the list of files used on validation')
    parser.add_argument('--classname', type=str, default='list/scenes_labels.txt',
                        help='str, path to the file that contains the name of the classes')
    parser.add_argument('--dataset', type=str, default='nyu_v2', help='str, name of the dataset used')
    parser.add_argument('--weights', type=str, default='',
                        help='str, path to the file that contains the weighs assigned to each class')
    parser.add_argument('--odir', type=str, default='./results/geometric3d',
                        help='str, path to the folder to be used to store the results')
    parser.add_argument('--exp_name', type=str, default='geometric3dbranch', help='str, name of the current experiment')
    parser.add_argument('--nfeatures', type=int, default='3', help='int, number of features used as input')
    parser.add_argument('--range01', default=False, action='store_true', help='bool, normalize features to 0-1 range')
    parser.add_argument('--pos_int16', default=False, action='store_true',
                        help='bool, positions are encoded with int16')

    parser.add_argument('--model_config', type=str, default='-',
                        help='str, defines the model as a sequence of layers')
    parser.add_argument('--seed', default=1, type=int, help='int, seed for random initialization. Default 1)')

    parser.add_argument('--fnet_widths', type=str,
                        default='[128]', help='str, list of widths of the hidden filter net layers (excluding the input and output ones, they are automatically calculated)')
    parser.add_argument('--fnet_llbias', default=False, action='store_true',
                        help='bool, use bias in the last layer in filter gen net')
    parser.add_argument('--fnet_tanh', default=False, action='store_true',
                        help='bool, use tanh as output activation in filter gen net.')
    parser.add_argument('--conv_bias', default=False, action='store_true', help='bool, use bias for convolutions')
    parser.add_argument('--edge_attr', type=str, default='posspherical-featureoffsets',
                        help='str, edge attributes. Default posspherical-featureoffsets')

    parser.add_argument('--resume', default=False, action='store_true',
                        help='bool, loads the last saved checkpoint of the same experiment')
    parser.add_argument('--transfer_learning', type=str, default='',
                        help='str, path to the weighs used to do the transfer learning')

    parser.add_argument('--pc_augm_input_dropout', type=float, default=0,
                        help='float, probability of removing points in input point clouds')
    parser.add_argument('--pc_augm_rot', default=False, action='store_true', help='bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', type=float, default=0.5,
                        help='float, probability of mirroring about x or y axis')
    parser.add_argument('--random_crop', default=False, action='store_true', help='bool, activate 3D random crop')
    parser.add_argument('--factor_rand', default=False, action='store_true',
                        help='bool, create crop with a random factor inside range [factor, 1]')
    parser.add_argument('--factor', type=float, default='0.875', help='float, crop factor. Default 0.875')

    args = parser.parse_args()
    args.fnet_widths = ast.literal_eval(args.fnet_widths)
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

    with open(os.path.join(exp_path, 'cmdline.txt'), 'w') as f:
        f.write(" ".join(sys.argv))

    train_writer = SummaryWriter(log_train_path)
    val_writer = SummaryWriter(log_val_path)

    features = args.nfeatures

    model = models.GraphNetwork(args.model_config, features, multigpu=args.multigpu,
                                default_fnet_widths=args.fnet_widths,
                                default_fnet_llbias=args.fnet_llbias,
                                default_edge_attr=args.edge_attr,
                                default_conv_bias=args.conv_bias,
                                default_fnet_tanh=args.fnet_tanh)

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
    assert args.batch_size % args.batch_parts == 0

    transform3d = {"dropout": args.pc_augm_input_dropout,
                   "rot": args.pc_augm_rot,
                   "mirror": args.pc_augm_mirror_prob}

    train_dataset = PCH5Dataset(args.dataset_path, args.dataset_folder,
                                args.train_split, transform3d=transform3d,
                                range01=args.range01,
                                pos_int16=args.pos_int16,
                                random_crop=args.random_crop,
                                factor_rand=args.factor_rand,
                                factor=args.factor)

    val_dataset = PCH5Dataset(args.dataset_path, args.dataset_folder,
                              args.val_split,
                              range01=args.range01,
                              pos_int16=args.pos_int16)

    train_loader = torch_geometric.data.DataLoader(train_dataset,
                                                   batch_size=int(args.batch_size/args.batch_parts),
                                                   num_workers=args.nworkers,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   pin_memory=False
                                                   )
    val_loader = torch_geometric.data.DataLoader(val_dataset,
                                                 batch_size=int(args.batch_size/args.batch_parts),
                                                 num_workers=args.nworkers,
                                                 shuffle=False,
                                                 pin_memory=False
                                                 )

    is_best_meanacc = False
    best_meanacc = 0
    start_epoch = 0
    resume_done = 0

    if args.resume:
        checkpoint_path_file = os.path.join(checkpoint_path, 'checkpoint_latest.pth.tar')
        if (os.path.isfile(checkpoint_path_file)):
            resume_done = 1

            epoch, best_meanacc, model_state, optimizer_state = utils.load_checkpoint(checkpoint_path_file)

            start_epoch = epoch + 1
            model.load_state_dict(model_state)

            if len(optimizer_state['state']) > 0:
                optimizer.load_state_dict(optimizer_state['optimizer'])
            else:
                print('There are problems with the optimizer state')
        else:
            print('Checkpoint does not exist, starting new trainning')

    if args.transfer_learning != '' and args.transfer_learning != '-' and resume_done == 0:
        if not os.path.isfile(args.transfer_learning):
            raise RuntimeError("Transfer learning model does not exist")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.transfer_learning)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        del pretrained_dict
        print("Loaded pretrained model")

    if start_epoch == (args.epochs-1):
        print('Training already finished, stopping job')
        exit()

    for epoch in range(start_epoch, args.epochs):

        print('Epoch {}/{} ({}):'.format(epoch, args.epochs, args.exp_name))

        train_loss, train_cm = train(model, train_loader,
                                     loss_criterion,
                                     optimizer,
                                     label_names,
                                     batch_parts=args.batch_parts,
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
