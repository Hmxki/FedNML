#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import math
import time
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


def max_m_indices(lst, n):
    elements_with_indices = list(enumerate(lst))
    sorted_elements = sorted(elements_with_indices, key=lambda x: x[1], reverse=True)
    top_n_elements = sorted_elements[:n]
    return [index for index, value in top_n_elements]


def min_n_indices(lst, n):
    elements_with_indices = list(enumerate(lst))
    sorted_elements = sorted(elements_with_indices, key=lambda x: x[1])
    bottom_n_elements = sorted_elements[:n]
    return [index for index, value in bottom_n_elements]


class CosineSimilarityFast(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarityFast, self).__init__()

    def forward(self, x1, x2):
        x2 = x2.t()
        # print(x1.shape)
        # print(x2)
        # print(x2.shape)
        # input()
        x = x1.mm(x2)

        x1_frobenius = x1.norm(dim=1).unsqueeze(0).t()
        x2_frobenins = x2.norm(dim=0).unsqueeze(0)
        x_frobenins = x1_frobenius.mm(x2_frobenins)

        final = x.mul(1/x_frobenins)
        final = torch.squeeze(final, dim=1)
        return final


class LogitAdjust_Multilabel(nn.Module):
    def __init__(self, cls_num_list, num, tau=1, weight=None):
        super(LogitAdjust_Multilabel, self).__init__()
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float32,device='cuda:0')
        self.cls_p_list = cls_num_list / num
        self.weight = weight

    def forward(self, x, target):
        x_m = x.clone()
        # for i in range(len(self.cls_p_list)): #abu1
        #     x_m[:, i] = (x_m[:, i]*self.cls_p_list[i])/(x_m[:, i]*self.cls_p_list[i] + (1-x_m[:, i])*(1-self.cls_p_list[i]))
        # nan_mask = torch.isnan(x_m)
        # x_m[nan_mask] = 0.
        return F.binary_cross_entropy(x_m, target, weight=self.weight, reduction='none')


class LA_KD(nn.Module):
    def __init__(self, cls_num_list, num, active_class_list_client, negative_class_list_client, tau=1, weight=None):
        super(LA_KD, self).__init__()
        self.active_class_list_client = active_class_list_client
        pos_prob = torch.as_tensor(cls_num_list, dtype=torch.float32)
        pos_prob = pos_prob/num
        neg_prob = 1 - pos_prob

        self.pos_m = tau * torch.log(pos_prob.clamp(min=1e-12))  # 正样本调整 [C]
        self.neg_m = tau * torch.log(neg_prob.clamp(min=1e-12))  # 负样本调整 [C]

        # cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        # self.active_class_list_client = active_class_list_client
        # self.negative_class_list_client = negative_class_list_client
        # self.cls_p_list = cls_num_list / num
        # self.weight = weight
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x, target, soft_target, w_kd):
        # Logit Adjustment
        adjustment = torch.where(target == 1, self.pos_m.to(x.device), self.neg_m.to(x.device)).to(x.device)
        adjusted_logits = x + adjustment

        # 多标签知识蒸馏损失（二元KL散度）
        student_probs = torch.sigmoid(adjusted_logits)
        kd_loss = F.binary_cross_entropy(
            student_probs,
            soft_target,
            reduction='mean'
        )

        # 多标签分类损失（带adjustment的BCE）
        cls_loss = F.binary_cross_entropy_with_logits(
            adjusted_logits,
            target.float(),
            reduction='mean'
        )


        return w_kd * kd_loss + (1 - w_kd) * cls_loss


class DatasetSplit_pseudo(Dataset):
    def __init__(self, dataset, idxs, client_id, args, active_class_list, negative_class_list, traindata_idx):
        self.dataset = dataset
        self.negative_class_list = negative_class_list
        self.idxs = list(idxs)
        self.client_id = client_id  # choose active classes
        self.annotation_num = args.annotation_num
        self.active_class_list = active_class_list
        self.traindata_idx = traindata_idx
        self.args = args
        self.idx_conf = []
        for i in range(len(traindata_idx) // 2):
            self.idx_conf += (traindata_idx[2 * i] + traindata_idx[2 * i + 1])
        self.idx_nconf = list(set(self.idxs) - set(self.idx_conf))
        #logging.info(f"Client ID: {self.client_id}, active_class_list: {self.active_class_list}")

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        distill_cls = torch.zeros(self.args.num_classes)
        sample = self.dataset[self.idxs[item]]
        for i in range(len(sample['target'])):
            if i not in self.active_class_list:
                sample['target'][i] = 0
        for i in range(len(self.traindata_idx)//2):
            idx0 = self.traindata_idx[2*i]
            idx1 = self.traindata_idx[2*i+1]
            if self.idxs[item] in (idx0+idx1):
                if self.idxs[item] in idx1:
                    sample['target'][self.negative_class_list[i]] = 1
            else:
                distill_cls[self.negative_class_list[i]] = 1
        return sample, self.idxs[item], distill_cls


    def get_num_of_each_class(self, args):
        class_sum = np.array([0.] * args.num_classes)
        for idx in self.idxs:
            class_sum += self.dataset.targets[idx]
        return class_sum.tolist()
    def mixup_data(self, x1, x2, alpha=1.0):
        '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.
        mixed_x = lam * x1 + (1 - lam) * x2
        return mixed_x, lam


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
        self.dataset = dataset
        self.pos_sample = pos_sample
        self.idxs = idxs
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

        self.loss_w = [len(self.trainloader.dataset) / i for i in self.class_num_list]

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

        trainloader = DataLoader(self.DS, batch_size=self.args.local_bs, shuffle=True,num_workers=2)
        return trainloader


    def find_indices_in_a(self, a, b):
        return torch.where(a.unsqueeze(0) == b.unsqueeze(1))[1]

    def train_FedMLP(self, rnd, tao, Prototype, net):    # my method
        # assert len(self.ldr_train.dataset) == len(self.idxs)
        # print(f"Client ID: {self.client_id}, Num: {len(self.ldr_train.dataset)}")
        if rnd < self.args.rounds_FedMLP_stage1:  # stage1

            glob_model = deepcopy(net)
            net.to(self.device)
            net.train()
            glob_model.to(self.device)
            glob_model.eval()
            # set the optimizer
            self.optimizer = torch.optim.AdamW(net.parameters(), lr=self.args.lr,
                                          weight_decay=1e-4)
            # train and update
            epoch_loss = []
            #print(self.loss_w)
            bce_criterion_sup = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.loss_w).cuda(),
                                                 reduction='none')  # include sigmoid
            bce_criterion_unsup = nn.MSELoss()
            for epoch in range(self.args.local_ep):
                print('client:', self.client_id, 'local_epoch:', epoch)
                batch_loss = []
                #batch_idx, (images, _, labels, items)
                #for j, (samples, item, active_class_list) in enumerate(self.trainloader):
                for j, (images, images2, labels, items) in enumerate(self.trainloader):
                    active_class_list_client = self.active_class_list
                    negetive_class_list_client = self.missing_list

                    criterion = LogitAdjust_Multilabel(cls_num_list=self.class_num_list, num=len(self.trainloader.dataset))
                    mse_loss = nn.MSELoss(reduction='none')
                    images1, images2, labels = images.to(self.device), images2.to(self.device), labels.to(self.device)
                    fe1 = logits1 = net(images1)
                    logits1_sig = torch.sigmoid(logits1).cuda()
                    logits2 = net(images2)
                    logits2_sig = torch.sigmoid(logits2).cuda()
                    # loss_sup1 = bce_criterion_sup(logits1, labels)  # tensor(32, 5)
                    # loss_sup2 = bce_criterion_sup(logits2, labels)  # tensor(32, 5)
                    with torch.no_grad():
                        outputs_global = glob_model(images1)
                        logits3 = torch.sigmoid(outputs_global).cuda()
                        outputs_global = glob_model(images2)
                        logits4 = torch.sigmoid(outputs_global).cuda()
                    loss_dis1 = mse_loss(logits1_sig, logits3).cuda()
                    loss_dis2 = mse_loss(logits2_sig, logits4).cuda()
                    loss_dis = (loss_dis1 + loss_dis2) / 2.
                    loss_sup1 = criterion(logits1_sig, labels)  # tensor(32, 5)
                    loss_sup2 = criterion(logits2_sig, labels)  # tensor(32, 5)
                    loss_sup = (loss_sup1 + loss_sup2) / 2.
                    # loss_sup = loss_sup.sum() / (self.args.batch_size * self.args.n_classes)  # supervised_loss

                    loss_sup = loss_sup[:, active_class_list_client].sum() / (
                                self.args.local_bs * self.args.annotation_num)  # supervised_loss
                    loss_dis = loss_dis[:, negetive_class_list_client].sum() / (
                                self.args.local_bs * len(negetive_class_list_client))  # supervised_loss

                    loss_unsup = bce_criterion_unsup(logits1_sig[:, negetive_class_list_client],
                                          logits2_sig[:, negetive_class_list_client])
                    loss = loss_sup + 0.0*loss_unsup + loss_dis
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    batch_loss.append(loss.item())
                epoch_loss.append(np.array(batch_loss).mean())
            if rnd == self.args.rounds_FedMLP_stage1 - 1:    # first tao and proto
                #print('client: ', self.client_id, active_class_list_client)
                proto = torch.zeros((self.args.num_classes * 2, len(fe1[0]))) # [cls0proto0, cls0proto1, cls1proto0...]
                # proto = np.array([torch.zeros_like(fe1[0].cpu())] * self.args.n_classes * 2)
                num_proto = [0] * self.args.num_classes * 2
                t = np.array([0] * self.args.num_classes)
                test_loader = DataLoader(dataset=self.DS, batch_size=self.args.local_bs, shuffle=False,num_workers=0)
                net.eval()
                with torch.no_grad():
                    for images, images2, labels, items in test_loader:
                        images1, labels = images.to(self.device), labels.to(self.device)
                        feature = outputs = net(images1)
                        probs = torch.sigmoid(outputs)  # soft predict
                        for cls in self.active_class_list:
                            idx0 = torch.where(labels[:, cls] == 0)[0].tolist()
                            idx1 = torch.where(labels[:, cls] == 1)[0].tolist()
                            num_proto[2*cls] += len(idx0)
                            num_proto[2*cls+1] += len(idx1)
                            proto[2*cls] = feature[idx0, :].sum(0).cpu() + proto[2*cls]
                            proto[2*cls+1] = feature[idx1, :].sum(0).cpu() + proto[2*cls+1]
                            # t[cls] += torch.sum(probs[idx0, cls] < self.args.L).item() + torch.sum(
                            #     probs[idx1, cls] > self.args.U).item()
                        for cls in self.missing_list:
                            t[cls] += torch.sum(
                                torch.logical_or(probs[:, cls] < self.args.L, probs[:, cls] > self.args.U)).item()
                for cls in self.active_class_list:
                    proto[2*cls] = proto[2*cls] / num_proto[2*cls]
                    proto[2*cls+1] = proto[2*cls+1] / num_proto[2*cls+1]
                t = t / len(self.DS)
                #print('local_t: ', t)
                return net.state_dict(), np.array(epoch_loss).mean(), negetive_class_list_client, active_class_list_client, t, proto
            else:
                return net.state_dict(), np.array(epoch_loss).mean(), negetive_class_list_client, active_class_list_client

        else:  # stage2
            #print(self.local_dataset.active_class_list)
            # find train samples for each class
            idx = []    # [[negcls1], [negcls2]]
            idxss = []
            feature = []
            similarity = []
            clean_idx = []
            noise_idx = []
            label = []  # [[negcls1], [negcls2]]
            num_train = 0
            glob_model = deepcopy(net)
            net.eval()
            glob_model.to(self.device)
            net.to(self.device)
            class_idx = torch.tensor([])
            l = torch.tensor([]).cuda()
            f = torch.tensor([]).cuda()
            t1 = time.time()
            if rnd == self.args.rounds_FedMLP_stage1:
                # print(len(self.ldr_train.dataset))
                self.traindata_idx = []  # [[negcls1_clean_train_idx], [negcls1_noise_train_idx], [negcls2_clean_train_idx], [negcls2_noise_train_idx]] idx
                for images, images2, labels, items in self.trainloader:
                    class_idx = torch.cat((class_idx, items), dim=0)
                    images1, labels = images.to(self.device), labels.to(self.device)
                    with torch.no_grad():
                        features = _ = net(images1)
                    f = torch.cat((f, features), dim=0)
                    l = torch.cat((l, labels), dim=0)
                for i in range(len(self.missing_list)):
                    feature.append(f)   # miss n classes
                    idx.append(class_idx)
                    label.append(l)
            else:
                for images, images2, labels, items in self.trainloader:
                    class_idx = torch.cat((class_idx, items), dim=0)
                    images1, labels = images.to(self.device), labels.to(self.device)
                    with torch.no_grad():
                        features = _ = net(images1)
                    f = torch.cat((f, features), dim=0)
                    l = torch.cat((l, labels), dim=0)
                for i in range(len(self.idxss)):
                    result_indices = self.find_indices_in_a(class_idx, torch.tensor(self.idxss[i]))
                    feature.append(f[result_indices])
                    idx.append(class_idx[result_indices])
                    label.append(l[result_indices])
            t2 = time.time()
            # print('feature_label_prepare_time: ', t2-t1)
            for i, cls in enumerate(self.missing_list):
                # sim = []
                proto_0 = Prototype[2*cls]
                proto_1 = Prototype[2*cls+1]
                model = CosineSimilarityFast().cuda()
                sim = (model(feature[i], torch.unsqueeze(proto_0.cuda(), dim=0)) - model(feature[i], torch.unsqueeze(proto_1.cuda(), dim=0))).tolist()
                similarity.append(sim)
            t3 = time.time()
            # print('sim_compute_time: ', t3 - t2)
            for i in range(len(self.missing_list)):
                idx_0 = np.where(np.array(similarity[i]) >= 0)[0]
                idx_1 = np.where(np.array(similarity[i]) < 0)[0]
                clean_idx.append(idx_0.tolist())
                noise_idx.append(idx_1.tolist())
            if rnd == self.args.rounds_FedMLP_stage1:
                for i, cls in enumerate(self.missing_list):
                    #print('cls', cls, 'tao: ', tao[cls])
                    num_clean_cls = int(1 * self.args.clean_threshold * len(clean_idx[i]))
                    num_noise_cls = int(1 * self.args.noise_threshold * len(noise_idx[i]))
                    # num_clean_cls = int(tao[cls] * len(clean_idx[i]))
                    # num_noise_cls = int(tao[cls] * len(noise_idx[i]))
                    num_train = num_train + num_noise_cls + num_clean_cls
                    max_m_indices_list = np.array(max_m_indices(similarity[i], num_clean_cls))
                    min_n_indices_list = np.array(min_n_indices(similarity[i], num_noise_cls))
                    if len(max_m_indices_list) == 0 and len(max_m_indices_list) == 0:
                        negcls_clean_train_idx = []
                        negcls_noise_train_idx = []
                    elif len(min_n_indices_list) == 0 and len(max_m_indices_list) != 0:
                        negcls_noise_train_idx = []
                        negcls_clean_train_idx = np.array(idx[i])[max_m_indices_list].tolist()
                    elif len(min_n_indices_list) != 0 and len(max_m_indices_list) == 0:
                        negcls_noise_train_idx = np.array(idx[i])[min_n_indices_list].tolist()
                        negcls_clean_train_idx = []
                    else:
                        negcls_clean_train_idx = np.array(idx[i])[max_m_indices_list].tolist()
                        negcls_noise_train_idx = np.array(idx[i])[min_n_indices_list].tolist()
                    self.traindata_idx.append(negcls_clean_train_idx)
                    self.traindata_idx.append(negcls_noise_train_idx)
            else:
                for i, cls in enumerate(self.missing_list):
                    #print('cls', cls, 'tao: ', tao[cls])
                    num_clean_cls = int(1 * self.args.clean_threshold * len(clean_idx[i]))
                    num_noise_cls = int(1 * self.args.noise_threshold * len(noise_idx[i]))
                    num_train = num_train + num_noise_cls + num_clean_cls
                    max_m_indices_list = np.array(max_m_indices(similarity[i], num_clean_cls))
                    min_n_indices_list = np.array(min_n_indices(similarity[i], num_noise_cls))

                    if len(max_m_indices_list) == 0 and len(max_m_indices_list) == 0:
                        negcls_clean_train_idx = []
                        negcls_noise_train_idx = []
                    elif len(min_n_indices_list) == 0 and len(max_m_indices_list) != 0:
                        negcls_noise_train_idx = []
                        negcls_clean_train_idx = np.array(idx[i])[max_m_indices_list].tolist()
                    elif len(min_n_indices_list) != 0 and len(max_m_indices_list) == 0:
                        negcls_noise_train_idx = np.array(idx[i])[min_n_indices_list].tolist()
                        negcls_clean_train_idx = []
                    else:
                        negcls_clean_train_idx = np.array(idx[i])[max_m_indices_list].tolist()
                        negcls_noise_train_idx = np.array(idx[i])[min_n_indices_list].tolist()
                    self.traindata_idx[2*i].extend(negcls_clean_train_idx)
                    self.traindata_idx[2*i+1].extend(negcls_noise_train_idx)

            t4 = time.time()
            # print('traindata_split_time: ', t4 - t3)

            for i, cls in enumerate(self.missing_list):
                #print('class: ', cls, 'clean_train_samples: ', len(self.traindata_idx[2*i]))
                #print('class: ', cls, 'noise_train_samples: ', len(self.traindata_idx[2 * i+1]))
                self.class_num_list[cls] = len(self.traindata_idx[2 * i+1])
            t5 = time.time()
            # print('acc_compute_time: ', t5 - t4)
            # train
            net.train()
            glob_model.eval()
            # set the optimizer
            self.optimizer = torch.optim.AdamW(net.parameters(), lr=self.args.lr,
                                          weight_decay=1e-4)
            # train and update
            epoch_loss = []
            loss_w = self.loss_w
            for i, cls in enumerate(self.missing_list):
                if len(self.traindata_idx[2*i+1]) != 0:
                    loss_w[cls] = len(self.traindata_idx[2*i]) / len(self.traindata_idx[2*i+1])
                else:
                    loss_w[cls] = 5.0
            #print(loss_w)
            bce_criterion_sup = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(loss_w).cuda(),
                                                     reduction='none')  # include sigmoid
            mse_loss = nn.MSELoss(reduction='none')

            for epoch in range(self.args.local_ep):
                print('client:', self.client_id, 'local_epoch:', epoch)
                batch_loss = []
                dataset = DatasetSplit_pseudo(self.dataset, self.idxs, self.client_id, self.args,
                                       self.active_class_list, self.missing_list, self.traindata_idx)
                dataloader = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True,
                                                 num_workers=0)
                for samples, item, distill_cls in dataloader:
                    distill_cls = distill_cls.cuda()
                    sup_cls = (~distill_cls.bool()).float().cuda()
                    criterion = LogitAdjust_Multilabel(cls_num_list=self.class_num_list,
                                                       num=len(self.idxs))
                    images1, images2, labels = samples["image_aug_1"].to(self.device), samples["image_aug_2"].to(self.device), samples["target"].to(
                            self.device)
                    feature = outputs = net(images1)
                    logits1 = torch.sigmoid(outputs).cuda()
                    with torch.no_grad():
                        _ = outputs_global = glob_model(images1)
                        logits2 = torch.sigmoid(outputs_global).cuda()
                    # loss_sup = bce_criterion_sup(outputs, labels).cuda()
                    loss_sup = criterion(logits1, labels).cuda()
                    loss_dis = mse_loss(logits1, logits2).cuda()

                    loss_noise = self.criterion_(outputs, logits2)
                    loss_noise = loss_noise[:, self.active_class_list]
                    loss_noise = loss_noise.mean()

                    loss = ((loss_sup * sup_cls).sum() + (loss_dis * distill_cls).sum()) / (sup_cls.sum() + distill_cls.sum())+loss_noise
                    #loss = (loss_sup * sup_cls).sum() / sup_cls.sum()
                    # print(loss)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    batch_loss.append(loss.item())

                epoch_loss.append(np.array(batch_loss).mean())
            for i in range(len(self.traindata_idx) // 2):
                idxs = self.traindata_idx[2 * i] + self.traindata_idx[2 * i + 1]
                idxss.append(idxs)
            t6 = time.time()
            # print('local_train_time: ', t6 - t5)
            self.idxss = idxss
            for i in range(len(idxss)):
                self.idxss[i] = list(set(self.idxs)-set(idxss[i]))  # [[negcls1_else_idx], [negcls2_else_idx]]

            # proto = np.array(
            #     [torch.zeros_like(feature[0].cpu())] * self.args.n_classes * 2)  # [cls0proto0, cls0proto1, cls1proto0...]
            proto = torch.zeros((self.args.num_classes * 2, len(feature[0])))  # [cls0proto0, cls0proto1, cls1proto0...]
            num_proto = [0] * self.args.num_classes * 2
            t = np.array([0] * self.args.num_classes)


            test_loader = DataLoader(dataset=self.DS, batch_size=self.args.local_bs, shuffle=False, num_workers=0)

            net.eval()
            with torch.no_grad():
                for images, images2, labels, items in test_loader:
                    images1, labels = images.to(self.device), labels.to(self.device)
                    feature = outputs = net(images1)
                    probs = torch.sigmoid(outputs)  # soft predict
                    for cls in self.active_class_list:
                        idx0 = torch.where(labels[:, cls] == 0)[0]
                        idx1 = torch.where(labels[:, cls] == 1)[0]
                        num_proto[2 * cls] += len(idx0)
                        num_proto[2 * cls + 1] += len(idx1)
                        proto[2 * cls] += feature[idx0, :].sum(0).cpu()
                        proto[2 * cls + 1] += feature[idx1, :].sum(0).cpu()
                        # t[cls] += torch.sum(probs[idx0, cls] < self.args.L).item() + torch.sum(
                        #     probs[idx1, cls] > self.args.U).item()
                    for cls in self.missing_list:
                        t[cls] += torch.sum(torch.logical_or(probs[:, cls] < self.args.L, probs[:, cls] > self.args.U)).item()
            for cls in self.active_class_list:
                if num_proto[2 * cls] == 0:
                    proto[2 * cls] = proto[2 * cls]
                else:
                    proto[2 * cls] = proto[2 * cls] / num_proto[2 * cls]
                if num_proto[2 * cls+1] == 0:
                    proto[2 * cls+1] = proto[2 * cls+1]
                else:
                    proto[2 * cls+1] = proto[2 * cls+1] / num_proto[2 * cls+1]
            t = t / len(self.DS)
            #print('local_t: ', t)
            net.cpu()
            self.optimizer.zero_grad()
            t7 = time.time()
            #print('local_test_proto_time: ', t7 - t6)
            return net.state_dict(), np.array(
                epoch_loss).mean(), self.missing_list, self.active_class_list, t, proto


    def update_weights(self, model, global_round,weight_kd):
        # Set mode to train model
        alpha, beta = self.sigmoid_weights_clipped(global_round, max_epoch=100, k=0.1, min_val=0.1, max_val=0.9)

        # self.correct_labels_with_Gmm(glob_model=deepcopy(model), w=[alpha, beta],
        #                                   global_round=global_round)

        glob_model =deepcopy(model)
        glob_model.eval()
        glob_model.to(self.device)
        model.train()
        model.to(self.device)
        epoch_loss = []

        # Set optimizer for the local updates

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr,
                                        weight_decay=1e-4)
        if global_round<10:
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, _, labels, items) in enumerate(self.trainloader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(images)

                    loss_ = self.criterion_(outputs, labels)
                    loss_ = loss_[:, self.active_class_list]
                    loss = loss_.mean()
                    # loss = loss_[:,self.active_class_list].mean()

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
            criterion = LA_KD(cls_num_list=self.class_num_list, num=len(self.trainloader.dataset),
                              active_class_list_client=self.active_class_list,
                              negative_class_list_client=[])
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, _, labels_raw, items) in enumerate(self.trainloader):
                    images = images.to(self.device)
                    # labels_raw = labels_raw.to(self.device)
                    labels = torch.tensor(self.dataset_ALR[items]).to(self.device)
                    labels = labels.unsqueeze(0) if labels.dim() == 1 else labels  # 统一为2维

                    with torch.no_grad():
                        logits0 = glob_model(images)
                        logist_glo_sig = torch.sigmoid(logits0.detach()).cuda()

                    outputs = model(images)
                    loss = criterion(outputs, labels, logist_glo_sig, weight_kd)

                    # loss_correction_1 = self.criterion(outputs['logits'], labels)
                    #
                    # loss_correction_2 = self.criterion(outputs['logits'], logist_glo_sig)
                    #
                    # loss = loss_correction_1 + loss_correction_2

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

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)









def test_inference(args, model, test_dataset, device):
    """ Returns the test accuracy and loss.
    """

    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()
    testloader = DataLoader(test_dataset, batch_size=32,
                            shuffle=False,num_workers=2)
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


