#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import math
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
import torch
import random

from sklearn.mixture import GaussianMixture
from torch import nn
import utils_.misc as misc
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, average_precision_score,
    hamming_loss, jaccard_score, recall_score, precision_score, cohen_kappa_score, multilabel_confusion_matrix
)
from sklearn.model_selection import KFold
from cleanlab.filter import _find_label_issues_multilabel


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_data(x, y, alpha=1.0, use_cuda=True):

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam



class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, client_id, args, pos_sample, active_class_list=None, ):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.client_id = client_id  # choose active classes
        self.annotation_num = args.annotation_num
        class_list = list(range(args.num_classes))
        if active_class_list is None:
            self.active_class_list = random.sample(class_list, self.annotation_num)
        else:
            self.active_class_list = active_class_list
        print(f"Client ID: {self.client_id}, active_class_list: {self.active_class_list}")
        self.pos_sample = pos_sample

    def __len__(self):
        return len(self.idxs)

    def get_num_of_each_class(self, args):
        class_sum = np.array([0.] * args.num_classes)
        for idx in self.idxs:
            label = self.dataset.targets[idx]
            class_sum = label+class_sum
        return class_sum.tolist()

    def __getitem__(self, item):
        sample = self.dataset[self.idxs[item]]
        for i in range(len(sample['target'])):
            if i not in self.active_class_list and self.idxs[item] in self.pos_sample[i]:
                sample['target'][i] = 0

        return sample['image_aug_1'], sample['image_aug_2'], sample['target'], self.idxs[item]


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, client_id, pos_sample, noise_idx_dict):
        self.args = args
        self.noise_idx = noise_idx_dict
        self.client_id = client_id
        self.logger = logger
        self.dataset_ALR = deepcopy(dataset).targets
        self.trainloader = self.train_val_test(dataset, list(idxs), pos_sample)
        self.class_num_list = self.DS.get_num_of_each_class(self.args)
        self.device = 'cuda'
        # Default criterion set to NLL loss function
        self.criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.criterion_ = torch.nn.BCEWithLogitsLoss(reduction='none').to(self.device)

        act_set = set(self.active_class_list)
        full_class_set = set(range(self.args.num_classes))
        miss_class_set = full_class_set - act_set
        self.missing_list = list(miss_class_set)
        label_mask = torch.zeros(self.args.num_classes)
        label_mask[list(miss_class_set)] = 1
        self.label_mask = label_mask.to(self.device)

        # GMM 建模
        self.gmm = GaussianMixture(n_components=2, random_state=0)

        self.num_work = [0, 0, 0, 0, 0, 0, 0, 0]
        self.batch_size = [16, 16, 16, 16, 16, 16, 16, 16]

    def train_val_test(self, dataset, idxs, pos_sample):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        self.DS = DatasetSplit(dataset, idxs, self.client_id, self.args, pos_sample, active_class_list=None)
        self.active_class_list = self.DS.active_class_list

        trainloader = DataLoader(self.DS, batch_size=self.args.local_bs, shuffle=True)
        return trainloader



    def sigmoid_weights_clipped(self, epoch, max_epoch, k=0.8, min_val=0.1, max_val=0.9):
        tau = 10
        beta_raw = 1 / (1 + np.exp(-k * (epoch - tau)))
        beta = beta_raw * (max_val - min_val) + min_val
        alpha = 1 - beta
        return alpha, beta


    def update_weights(self, model, global_round, w_g, mu):
        # Set mode to train model

        net_glob = w_g
        net_glob.to(self.device)
        model.train()
        model.to(self.device)
        epoch_loss = []

        # Set optimizer for the local updates

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr,
                                        weight_decay=1e-4)
        if global_round<10000:
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, _, labels, items) in enumerate(self.trainloader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(images)

                    # with torch.no_grad():
                    #     logits0 = net_glob(images)
                    #     pseudo_label = torch.sigmoid(logits0.detach()).cuda()

                    loss0 = self.criterion_(outputs, labels)
                    loss_ = loss0[:, self.active_class_list]

                    # loss_missing = self.criterion_(outputs, pseudo_label)
                    # loss_missing = loss_missing[:, self.label_mask.cpu().numpy()]
                    # loss_missing = loss_missing.mean()

                    loss = loss_.mean()
                    #loss = loss_.mean()+loss_missing
                    # loss = loss_[:,self.active_class_list].mean()

                    if batch_idx == 0:
                        output_whole = np.array(outputs.detach().cpu())
                        loss_whole = np.array(loss0.mean(dim=1).detach().cpu())
                    else:
                        output_whole = np.concatenate((output_whole, outputs.detach().cpu()), axis=0)
                        loss_whole = np.concatenate((loss_whole, loss0.mean(dim=1).detach().cpu()), axis=0)


                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if self.args.verbose and (batch_idx % 10 == 0):
                        print(
                            '| Global Round : {} | client_id: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                global_round, self.client_id, iter, batch_idx * len(images),
                                len(self.trainloader.dataset),
                                                                    100. * batch_idx / len(self.trainloader),
                                loss.item()))
                    self.logger.add_scalar('loss', loss.item())
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        else:

            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, _, labels_raw, items) in enumerate(self.trainloader):
                    images = images.to(self.device)
                    # labels_raw = labels_raw.to(self.device)
                    labels = torch.tensor(self.dataset_ALR[items]).to(self.device)
                    labels = labels.unsqueeze(0) if labels.dim() == 1 else labels  # 统一为2维

                    log_probs = model(images)['logits']
                    outputs = torch.sigmoid(log_probs)
                    loss0 = self.criterion_(log_probs, labels)
                    loss = loss0.mean()

                    if self.args.beta > 0:
                        if batch_idx > 0:
                            w_diff = torch.tensor(0.).to(self.device)
                            for w, w_t in zip(net_glob.parameters(), model.parameters()):
                                w_diff += torch.pow(torch.norm(w - w_t), 2)
                            w_diff = torch.sqrt(w_diff)
                            loss += self.args.beta * mu * w_diff

                    if batch_idx == 0:
                        output_whole = np.array(outputs.detach().cpu())
                        loss_whole = np.array(loss0.mean(dim=1).detach().cpu())
                    else:
                        output_whole = np.concatenate((output_whole, outputs.detach().cpu()), axis=0)
                        loss_whole = np.concatenate((loss_whole, loss0.mean(dim=1).detach().cpu()), axis=0)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if self.args.verbose and (batch_idx % 10 == 0):
                        print(
                            '| Global Round : {} | client_id: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                global_round, self.client_id, iter, batch_idx * len(images),
                                len(self.trainloader.dataset),
                                                                    100. * batch_idx / len(self.trainloader),
                                loss.item()))
                    self.logger.add_scalar('loss', loss.item())
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), output_whole, loss_whole









def test_inference(args, model, test_dataset, device):
    """ Returns the test accuracy and loss.
    """

    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()
    testloader = DataLoader(test_dataset, batch_size=32,
                            shuffle=False)
    metric_logger = misc.MetricLogger(delimiter="  ")
    true_labels_all, pred_labels, pred_probs = [], [], []

    for samples in testloader:
        images, labels = samples['image'].to(device), samples['target'].to(device)
        labels = labels.unsqueeze(0) if labels.dim() == 1 else labels

        # Inference
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Prediction
        output_prob = torch.sigmoid(outputs)
        output_pred = (output_prob > 0.5).float()

        metric_logger.update(loss=loss.item())
        true_labels_all.extend(labels.cpu().numpy())
        pred_probs.extend(output_prob.detach().cpu().numpy())
        pred_labels.extend(output_pred.detach().cpu().numpy())

    all_labels = np.array(true_labels_all)
    all_probs = np.array(pred_probs)
    all_preds = np.array(pred_labels)

    print('pred', np.sum(all_preds, axis=0))
    print('labels', np.sum(all_labels, axis=0))

    APs = []

    for label_index in range(all_labels.shape[1]):
        true_labels = all_labels[:, label_index]
        predicted_scores = all_probs[:, label_index]
        ap = average_precision_score(true_labels, predicted_scores)
        APs.append(ap)

    mAP = torch.tensor(APs).mean()
    hamming_loss_ = hamming_loss(all_labels, all_preds)

    conf_matrices = multilabel_confusion_matrix(all_labels, all_preds)
    # 计算宏平均精确率、召回率、F1分数
    macro_precision = []
    macro_recall = []
    macro_f1 = []
    macro_BACC = []

    for cm in conf_matrices:
        TN = cm[0, 0]
        TP = cm[1, 1]
        FP = cm[0, 1]
        FN = cm[1, 0]

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        recall0 = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        macro_precision.append(precision)
        macro_recall.append(recall)
        macro_f1.append(f1)
        bacc = (recall0 + recall) / 2
        macro_BACC.append(bacc)

    P = np.mean(macro_precision)
    R = np.mean(macro_recall)
    F1 = np.mean(macro_f1)
    bacc = np.mean(macro_BACC)

    score = (mAP + P + F1 + bacc + R) / 5
    for idx, bacc_value in enumerate(macro_BACC):
        print(f"Class {idx}: BACC = {bacc_value:.4f}")

    print(f'val loss: {metric_logger.meters["loss"].global_avg}')
    print(f'map: {mAP:.4f}, F1 Score: {F1:.4f}, Hamming Loss: {hamming_loss_:.4f},\n'
          f' Precision: {P:.4f}, Recall: {R:.4f},\n'
          f' bacc: {bacc:.4f}, Score: {score:.4f}')

    metric_logger.synchronize_between_processes()
    return score


