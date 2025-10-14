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


class LA_KD(nn.Module):
    def __init__(self, cls_num_list, num, active_class_list_client, negative_class_list_client, tau=1, weight=None):
        super(LA_KD, self).__init__()

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

    def correct_labels_with_Gmm(self, glob_model, w, global_round, clean_interval=5):


        if global_round % clean_interval != 0 or global_round<10:
            return

        print(f"[Client {self.client_id}] Correcting labels using Gmm...")

        all_probs = []
        all_labels = []
        all_indices = []

        glob_model.eval()
        glob_model.to(self.device)

        with torch.no_grad():
            for batch_idx, (images_w, image_s, labels_, items) in enumerate(self.trainloader):
                images_w = images_w.to(self.device)
                #labels_ = labels_.to(self.device)
                labels = torch.tensor(self.dataset_ALR[items]).float().to(self.device)  # 真实标签 [N, C]
                #labels = w[0]*labels_+w[1]*labels_c
                logits = glob_model(images_w)['logits']
                probs = torch.sigmoid(logits)  # 预测概率 [N, C]

                all_probs.append(probs.detach().cpu())
                all_labels.append(labels.detach().cpu())
                all_indices.extend(items)  # 原始样本索引

        all_probs = torch.cat(all_probs, dim=0)  # [N, C]
        all_labels = torch.cat(all_labels, dim=0)  # [N, C]

        # 方法 A: 一致性得分
        consistency = (all_labels * all_probs + (1 - all_labels) * (1 - all_probs)).mean(dim=1)

        # 方法 B: BCE损失
        loss = F.binary_cross_entropy(all_probs, all_labels, reduction='none').mean(dim=1)
        neg_loss = -loss

        features = torch.stack([consistency, neg_loss], dim=1).numpy()  # Nx2

        # GMM 建模
        #gmm = GaussianMixture(n_components=2, random_state=0)
        self.gmm.fit(features)
        preds = self.gmm.predict(features)
        probs_gmm = self.gmm.predict_proba(features)

        means = self.gmm.means_[:, 0]
        noisy_cluster = np.argmin(means)
        noisy_indices = np.where(preds == noisy_cluster)[0]

        # 获取对应全局索引
        noisy_item_ids = [all_indices[i] for i in noisy_indices]

        print(f"[Client {self.client_id}] Detected {len(noisy_item_ids)} noisy samples.")

        # === 可选操作：刷新标签 ===
        # 例如将这些样本的标签更新为模型预测（软标签），而非剔除：
        sum_ = 0
        for i in noisy_item_ids:
            arr = all_probs[all_indices.index(i)].numpy()
            arr = np.where(arr > 0.9, 1, np.where(arr < 0.1, 0, arr))
            #sum_+=arr.sum()
            self.dataset_ALR[i] = arr
        #print(f"[Client {self.client_id}] Corrected {sum_} noisy labels.")

    def sigmoid_weights_clipped(self, epoch, max_epoch, k=0.8, min_val=0.1, max_val=0.9):
        tau = 10
        beta_raw = 1 / (1 + np.exp(-k * (epoch - tau)))
        beta = beta_raw * (max_val - min_val) + min_val
        alpha = 1 - beta
        return alpha, beta

    def get_multi_label_smoothed_distribution(self, labels, num_class, epsilon_pos=0.1, epsilon_neg=0.0):
        """
        Args:
            labels: (batch_size, num_class) 的多标签二进制矩阵
            num_class: 类别总数
            epsilon_pos: 正标签的平滑系数（降低正标签的置信度）
            epsilon_neg: 负标签的平滑系数（增加负标签的微小概率）
        Returns:
            smoothed_label: (batch_size, num_class) 的平滑后概率分布
        """
        device = labels.device
        smoothed_label = torch.zeros_like(labels, dtype=torch.float32)

        # 对正标签处理（1 -> 1-epsilon_pos）
        pos_mask = labels == 1
        smoothed_label[pos_mask] = 1.0 - epsilon_pos

        # 对负标签处理（0 -> epsilon_neg/(num_pos+1)）
        neg_mask = labels == 0
        num_pos = pos_mask.sum(dim=1, keepdim=True)  # 每个样本的正标签数
        neg_smooth_value = epsilon_neg / (num_pos + 1)  # 动态调整负标签平滑量
        smoothed_label[neg_mask] = neg_smooth_value.expand_as(labels)[neg_mask]

        # 归一化（可选，根据损失函数需求决定）
        # smoothed_label = smoothed_label / smoothed_label.sum(dim=1, keepdim=True)

        return smoothed_label.to(device)


    def update_weights(self, model, global_round,weight_kd):
        # Set mode to train model
        alpha, beta = self.sigmoid_weights_clipped(global_round, max_epoch=100, k=0.1, min_val=0.1, max_val=0.9)

        self.correct_labels_with_Gmm(glob_model=deepcopy(model), w=[alpha, beta],
                                          global_round=global_round)

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

                    loss_ = self.criterion_(outputs['logits'], labels)

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

                    with torch.no_grad():
                        logits0 = glob_model(images)
                        logist_glo_sig = torch.sigmoid(logits0['logits'].detach()).cuda()

                    outputs = model(images)
                    loss = criterion(outputs['logits'], labels, logist_glo_sig, weight_kd)

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
                            shuffle=False)
    metric_logger = misc.MetricLogger(delimiter="  ")
    true_labels_all, pred_labels, pred_probs = [], [], []

    for samples in testloader:
        images, labels = samples['image'].to(device), samples['target'].to(device)

        # Inference
        with torch.no_grad():
            outputs = model(images)['logits']
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

    print(f'val loss: {metric_logger.meters["loss"].global_avg}')
    print(f'map: {mAP:.4f}, F1 Score: {F1:.4f}, Hamming Loss: {hamming_loss_:.4f},\n'
          f' Precision: {P:.4f}, Recall: {R:.4f},\n'
          f' bacc: {bacc:.4f}, Score: {score:.4f}')

    metric_logger.synchronize_between_processes()
    return score


