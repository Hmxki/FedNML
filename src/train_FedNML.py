#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import sys
import time
import pickle
import numpy as np
from timm.layers import trunc_normal_
from tqdm import tqdm
from numpy import where
import torch
from tensorboardX import SummaryWriter
import random
from options import args_parser
from update_ours_all import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, Logger
from utils_.all_models import get_model, modify_last_layer
from utils_.noise import add_noise
from src import models_vit
from utils_.pos_embed import interpolate_pos_embed


def sigmoid_rampup(current, begin, end):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    current = np.clip(current, begin, end)
    phase = 1.0 - (current-begin) / (end-begin)
    return float(np.exp(-2.0 * phase * phase))
def get_current_consistency_weight(rnd, begin, end):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return sigmoid_rampup(rnd, begin, end)

if __name__ == '__main__':
    log_file = os.path.join("../output_print/brazilian", "time_test_2epoch-ours.txt")
    sys.stdout = Logger(log_file)

    start_time = time.time()
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu_id:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu_id else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # # --------------------- Add Noise ---------------------------
    max_1 = np.max(np.sum(train_dataset.targets, axis=1))
    y_train = np.array(train_dataset.targets)
    y_train_noisy, gamma_s, real_noise_level, noise_idx_dict = add_noise(
        args, y_train, user_groups, max_1=max_1)
    train_dataset.targets = y_train_noisy

    row_idx_1, column_idx_1 = where(train_dataset.targets == 1)
    pos_sample = []
    for i in range(args.num_classes):
        pos_sample.append(row_idx_1[where(column_idx_1 == i)[0]])


    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    elif args.model == 'ResNet':
        global_model = get_model('Resnet18',pretrained=True)
        global_model, _ = modify_last_layer('Resnet18', global_model, args.num_classes)

    elif args.model == 'RETFound_cfp':
        # call the model
        global_model = models_vit.__dict__['vit_large_patch16'](
            num_classes=args.num_classes,
            drop_path_rate=0.2,
            global_pool=True,
        )
        # load RETFound weights
        checkpoint = torch.load(
            '../RETFound_mae_natureCFP.pth',
            map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = global_model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(global_model, checkpoint_model)
        # load pre-trained model
        global_model.load_state_dict(checkpoint_model, strict=False)


    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model = global_model.to(device)
    #print(global_model)


    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    score = 0

    # checkpoint = torch.load("../output_print/global_model_resnet.pth", map_location=device)
    # global_model.load_state_dict(checkpoint)
    idxs_users = [i for i in range(args.num_users)]
    local_models = []
    for idx in idxs_users:
        local_models.append(LocalUpdate(args=args, dataset=train_dataset,
                                  idxs=user_groups[idx], logger=logger, client_id=idx, pos_sample=pos_sample,
                                  noise_idx_dict=noise_idx_dict[idx]))

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        #m = max(int(args.frac * args.num_users), 1)

        a = []
        weight_kd = get_current_consistency_weight(
            epoch, 10, 100) * 0.8

        # for i in range(100):
        #     weight_kd = get_current_consistency_weight(
        #         i, 10, 100) * 0.8
        #     a.append(weight_kd)
        # print(a)
        for idx in idxs_users:
            local_model = local_models[idx]
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, weight_kd=weight_kd)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)


        # test
        score_this = test_inference(args, global_model, test_dataset, device)
        if score_this>score:
            score = score_this
            print('best score:',score,'best epoch:',epoch)
            torch.save(global_model.state_dict(), '../output_print/global_model_RETFound_cfp.pth')

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

