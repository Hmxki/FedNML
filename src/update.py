#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import math
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
import torch
import random
from torch import nn
import utils_.lr_decay as lrd
import utils_.misc as misc
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, average_precision_score,
    hamming_loss, jaccard_score, recall_score, precision_score, cohen_kappa_score, multilabel_confusion_matrix
)


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

    def __getitem__(self, item):
        sample = self.dataset[self.idxs[item]]
        for i in range(len(sample['target'])):
            if i not in self.active_class_list and self.idxs[item] in self.pos_sample[i]:
                sample['target'][i] = 0

        return sample['image_aug_1'], sample['image_aug_2'], sample['target'], self.idxs[item]


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, client_id, pos_sample):
        self.args = args
        self.client_id = client_id
        self.logger = logger
        self.dataset_ALR = deepcopy(dataset).targets
        self.trainloader = self.train_val_test(dataset, list(idxs), pos_sample)
        self.device = 'cuda'
        # Default criterion set to NLL loss function
        self.criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.criterion_ = torch.nn.BCEWithLogitsLoss(reduction='none').to(self.device)
        self.aux_loss_func = F.l1_loss


        act_set = set(self.active_class_list)
        full_class_set = set(range(self.args.num_classes))
        miss_class_set = full_class_set - act_set
        self.missing_list = list(miss_class_set)
        label_mask = torch.zeros(self.args.num_classes)
        label_mask[list(miss_class_set)] = 1
        self.label_mask = label_mask.to(self.device)

        self.num_work = [0, 0, 0, 0, 0, 0, 0, 0]
        self.batch_size = [16, 16, 16, 16, 16, 16, 16, 16]

    def train_val_test(self, dataset, idxs, pos_sample):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        # idxs_train = idxs[:int(0.8*len(idxs))]
        # idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        # idxs_test = idxs[int(0.9*len(idxs)):]
        self.DS = DatasetSplit(dataset, idxs, self.client_id, self.args, pos_sample, active_class_list=None)
        self.active_class_list = self.DS.active_class_list

        trainloader = DataLoader(self.DS, batch_size=self.args.local_bs, shuffle=True)
        # validloader = DataLoader(DatasetSplit(dataset, idxs_val),
        #                          batch_size=int(len(idxs_val)/10), shuffle=False)
        # testloader = DataLoader(DatasetSplit(dataset, idxs_test),
        #                         batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader


    def compute_entropy(self, net=None, dataset=None):
        net.to(self.device)
        net.eval()
        #print('compute entropy')
        rank_array = torch.empty((0), device=self.device)
        with torch.no_grad():
            for images, _, _, _ in dataset:
                images = images.to(self.device)

                # label_mask:本地中的未知类即缺失标签类
                output = torch.sigmoid(net(images)['logits'].detach())
                output1 = output * self.label_mask

                entr_res = torch.special.entr(output1)
                ensemble_entropy = torch.mean(entr_res, dim=1)
                rank_array = torch.cat((rank_array, ensemble_entropy))

        # rank
        # indices : indices of ndarray rank_array, entropy value of those rows are k largest
        _, uncertain_indices = torch.topk(rank_array, k = math.floor(self.args.uncertain_pool_size*len(dataset.dataset)), largest=True)
        _, confident_indices = torch.topk(rank_array, k = math.floor(self.args.confident_pool_size*len(dataset.dataset)), largest=False)
        #print('compute finished!')
        return uncertain_indices.tolist(), confident_indices.tolist()

    def multilabel_entropy_loss(self, logits, reduction='mean'):
        probs = torch.sigmoid(logits)
        entropy = -probs * torch.log(probs + 1e-6) - (1 - probs) * torch.log(1 - probs + 1e-6)  # [N, C]
        losses = torch.mean(entropy, dim=1)  # 每个样本对多个标签的总熵
        if reduction == 'none':
            return losses
        elif reduction == 'mean':
            return torch.mean(losses)
        elif reduction == 'sum':
            return torch.sum(losses)

    def binary_kl_div(self, p, q, eps=1e-6):
        """
        p, q: tensor of shape [batch_size, num_classes], representing Bernoulli probabilities
        Compute KL divergence for multi-label classification
        """
        p = torch.clamp(p, eps, 1 - eps)
        q = torch.clamp(q, eps, 1 - eps)
        return (p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))).sum(dim=1)  # shape [batch_size]

    def symmetric_kl_div(self, p, q, base=2):
        return (self.binary_kl_div(p, q) + self.binary_kl_div(q, p))/2

    def js_div(self, p, q, base=2):
        # Jensen-Shannon divergence, value is in (0, 1)
        m = 0.5 * (p + q)
        return 0.5 * self.binary_kl_div(p, m) + 0.5 * self.binary_kl_div(q, m)


    def dynamic_dual_thresholds(self, pseudo_label, epoch, default_n=2.0, skewness_sensitivity=0.5,
                            base_ratio=0.9, max_epochs=200):
        # 计算每个类别的统计量
        mean = pseudo_label.mean(dim=0)  # [8]
        std = pseudo_label.std(dim=0)  # [8]
        skewness = torch.mean(((pseudo_label - mean) / std) ** 3, dim=0)  # [8]
        alpha = base_ratio ** (epoch / max_epochs )  # 乘以10加速衰减

        # 动态调整n值（偏度越大，调整幅度越大）
        n_values = torch.clamp(
            default_n - skewness_sensitivity * skewness.abs(),
            min=1.0,  # 保证n≥1
            max=3.0  # 避免n过大
        )

        # 计算双阈值
        upper_thresholds = (mean + n_values * std)*(2-alpha)
        lower_thresholds = (mean - n_values * std)*alpha

        return upper_thresholds, lower_thresholds

    def dynamic_thresholds(self, pseudo_label, epoch, default_n=2.0, skewness_sensitivity=0.5,
                            base_ratio=0.9, max_epochs=200):
        # 计算每个类别的统计量
        mean = pseudo_label.mean()  # [8]
        std = pseudo_label.std()  # [8]
        skewness = torch.mean(((pseudo_label - mean) / std) ** 3, dim=0)  # [8]
        alpha = base_ratio ** (epoch / max_epochs )  # 乘以10加速衰减
        n_values = torch.clamp(
            default_n - skewness_sensitivity * skewness.abs(),
            min=1.0,  # 保证n≥1
            max=3.0  # 避免n过大
        )

        # 计算双阈值
        upper_thresholds = (mean + n_values * std)*(2-alpha)

        return upper_thresholds

    def sigmoid_weights_clipped(self, epoch, max_epoch, k=0.8, min_val=0.1, max_val=0.9):
        tau = 10
        beta_raw = 1 / (1 + np.exp(-k * (epoch - tau)))
        beta = beta_raw * (max_val - min_val) + min_val
        alpha = 1 - beta
        return alpha, beta

    def _mixup(self, img1, img2, l1, l2):
        alpha = 0.2
        beta_dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))
        weight = beta_dist.sample()
        weight = torch.FloatTensor(weight)
        weight = weight.to(device=self.device)
        img =  weight * img1 + (1-weight)*img2
        label = weight * l1 + (1-weight)*l2
        return img, label


    def update_weights(self, model, global_round):
        # Set mode to train model
        glob_model =deepcopy(model)
        glob_model.eval()
        glob_model.to(self.device)
        model.train()
        model.to(self.device)
        # model_without_ddp = model
        #
        # no_weight_decay = model_without_ddp.no_weight_decay() if hasattr(model_without_ddp, 'no_weight_decay') else []
        # param_groups = lrd.param_groups_lrd(model_without_ddp, 1e-4,
        #                                     no_weight_decay,
        #                                     layer_decay=0.65)


        epoch_loss = []

        # Set optimizer for the local updates

       #optimizer = torch.optim.AdamW(param_groups, lr=self.args.lr)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr,
                                      weight_decay=1e-4)

        # stage1:
        if global_round<200:
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, _, labels, items) in enumerate(self.trainloader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(images)

                    loss_ = self.criterion_(outputs, labels)

                    loss = loss_.mean()
                    #loss = loss_[:,self.active_class_list].mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if self.args.verbose and (batch_idx % 10 == 0):
                        print('| Global Round : {} | client_id: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_round, self.client_id, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader), loss.item()))
                    self.logger.add_scalar('loss', loss.item())
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

            return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

        else:
            alpha, beta = self.sigmoid_weights_clipped(global_round, max_epoch=100, k=0.1, min_val=0.2, max_val=0.8)
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images_w, image_s, labels_, items) in enumerate(self.trainloader):
                    images_w = images_w.to(self.device)
                    image_s = image_s.to(self.device)
                    labels_raw = deepcopy(labels_).to(self.device)

                    # 校正缺失标签
                    with torch.no_grad():
                        logits0 = glob_model(images_w)
                        logist_glo_sig = torch.sigmoid(logits0['logits'].detach()).cuda()

                    # noise_indices
                    logits1 = model(image_s)
                    loss_correction = self.criterion_(logits1['logits'], logist_glo_sig)
                    loss_correction_ = loss_correction.mean()

                    loss_raw = self.criterion_(logits1['logits'], labels_raw)
                    loss_raw_ = loss_raw.mean()

                    loss = beta * loss_correction_ + alpha * loss_raw_

                    # loss = loss_correction

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


                    if self.args.verbose and (batch_idx % 10 == 0):
                        print('| Global Round : {} | client_id: {} | mid&confident data | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_round, self.client_id, iter, batch_idx * len(image_s),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader), loss.item()))
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

    print(f'val loss: {metric_logger.meters["loss"].global_avg}')
    print(f'map: {mAP:.4f}, F1 Score: {F1:.4f}, Hamming Loss: {hamming_loss_:.4f},\n'
          f' Precision: {P:.4f}, Recall: {R:.4f},\n'
          f' bacc: {bacc:.4f}, Score: {score:.4f}')

    metric_logger.synchronize_between_processes()
    return score


