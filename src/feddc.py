#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import sys
import time
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from timm.layers import trunc_normal_
from scipy.spatial.distance import cdist
from torch.utils.data import Subset
from tqdm import tqdm
from numpy import where
import torch
from tensorboardX import SummaryWriter
import random
from options import args_parser
from update_dc import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, Logger
from utils_.all_models import get_model, modify_last_layer
from utils_.noise import add_noise
from src import models_vit
from utils_.pos_embed import interpolate_pos_embed
import torch.nn.functional as F

def sigmoid_rampup(current, begin, end):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    current = np.clip(current, begin, end)
    phase = 1.0 - (current-begin) / (end-begin)
    return float(np.exp(-2.0 * phase * phase))
def get_current_consistency_weight(rnd, begin, end):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return sigmoid_rampup(rnd, begin, end)

def lid_term(X, batch, k=20):
    eps = 1e-6
    X = np.asarray(X, dtype=np.float32)

    batch = np.asarray(batch, dtype=np.float32)
    f = lambda v: - k / (np.sum(np.log(v / (v[-1]+eps)))+eps)
    distances = cdist(X, batch)

    # get the closest k neighbours
    sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=distances)[:, 1:k + 1]
    m, n = sort_indices.shape
    idx = np.ogrid[:m, :n]
    idx[1] = sort_indices
    # sorted matrix
    distances_ = distances[tuple(idx)]
    lids = np.apply_along_axis(f, axis=1, arr=distances_)
    return lids


def get_output(loader, net, args, latent=False, criterion=None):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, samples in enumerate(loader):
            images = samples['image_aug_1'].to('cuda')
            labels = samples['target'].to('cuda')
            # Converting the labels to long type.
            if latent == False:
                outputs = net(images)
                outputs = F.sigmoid(outputs)
            else:
                outputs = net(images, True)
            loss = criterion(outputs, labels).mean(dim=1)
            if i == 0:
                output_whole = np.array(outputs.cpu())
                loss_whole = np.array(loss.cpu())
            else:
                output_whole = np.concatenate((output_whole, outputs.cpu()), axis=0)
                loss_whole = np.concatenate((loss_whole, loss.cpu()), axis=0)
    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole


if __name__ == '__main__':

    log_file = os.path.join("../output_print/brazilian", "4-6-classbacc-feddc.txt")
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
    device = 'cuda'

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

        net_local = get_model('Resnet18', pretrained=True)
        net_local, _ = modify_last_layer('Resnet18', net_local, args.num_classes)

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

    LID_accumulative_client = np.zeros(args.num_users)

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        #m = max(int(args.frac * args.num_users), 1)

        a = []
        weight_kd = get_current_consistency_weight(
            epoch, 10, 100) * 0.8

        LID_whole = np.zeros(len(y_train_noisy))
        loss_whole = np.zeros(len(y_train_noisy))
        LID_client = np.zeros(args.num_users)
        loss_accumulative_whole = np.zeros(len(y_train_noisy))

        if epoch==0:
            mu_list = np.zeros(args.num_users)
        else:
            mu_list = estimated_noisy_level



        for idx in idxs_users:
            local_model = local_models[idx]
            mu_i = mu_list[idx]
            #net_local.load_state_dict(global_model.state_dict())

            w, loss, local_output, loss_per_sample = local_model.update_weights(model=copy.deepcopy(global_model), w_g=global_model, global_round=epoch, mu=mu_i)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            #net_local.load_state_dict(copy.deepcopy(w))

            LID_local = list(lid_term(local_output, local_output))
            LID_whole[list(user_groups[idx])] = LID_local
            loss_whole[list(user_groups[idx])] = loss_per_sample
            LID_client[idx] = np.mean(LID_local)


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



        LID_accumulative_client = LID_accumulative_client + np.array(LID_client)
        loss_accumulative_whole = loss_accumulative_whole + np.array(loss_whole)
        LID_accumulative_old = LID_accumulative_client.copy()

        # ----------------------Apply Gaussian Mixture Model to LID----------------------
        noisy_set = idxs_users
        clean_set = []

        estimated_noisy_level = np.zeros(args.num_users)

        for client_id in noisy_set:
            sample_idx = np.array(list(user_groups[client_id]))
            loss = np.array(loss_accumulative_whole[sample_idx])
            gmm_loss = GaussianMixture(n_components=2, random_state=args.seed).fit(np.array(loss).reshape(-1, 1))
            labels_loss = gmm_loss.predict(np.array(loss).reshape(-1, 1))
            gmm_clean_label_loss = np.argsort(gmm_loss.means_[:, 0])[0]

            pred_n = np.where(labels_loss.flatten() != gmm_clean_label_loss)[0]
            estimated_noisy_level[client_id] = len(pred_n) / len(sample_idx)
            #y_train_noisy_new = np.array(train_dataset.targets)

        if epoch >= 10:
            if args.correction:
                criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
                for idx in noisy_set:
                    sample_idx = np.array(list(user_groups[idx]))
                    dataset_client = Subset(train_dataset, sample_idx)
                    loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=args.local_bs, shuffle=False)
                    loss = np.array(loss_accumulative_whole[sample_idx])
                    local_output, _ = get_output(loader, global_model.to('cuda'), args, False, criterion)
                    relabel_idx = (-loss).argsort()[:int(len(sample_idx) * estimated_noisy_level[idx] * args.relabel_ratio)]
                    relabel_idx = list(
                        set(np.where(np.max(local_output, axis=1) > args.confidence_thres)[0]) & set(relabel_idx))

                    y_train_noisy_new = np.array(train_dataset.targets)
                    y_train_noisy_new[sample_idx[relabel_idx]] = local_output[relabel_idx]
                    train_dataset.targets = y_train_noisy_new





    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

