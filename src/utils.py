#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, MuReD_iid, MuReD_noniid_unequal, MuReD_noniid
from sampling import cifar_iid, cifar_noniid

class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        pass


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == "MuReD":
        #root = "../data/MuReD"
        root = "../data/46"
        #root = "../data/Brazilian"

        # mured
        # normalize = transforms.Normalize([0.509, 0.298, 0.144],
        #                                  [0.286, 0.184, 0.129])


        # 46
        normalize = transforms.Normalize([0.099, 0.190, 0.312],
                                         [0.101, 0.187, 0.298])

        # brazilian
        # normalize = transforms.Normalize([0.111, 0.299, 0.590],
        #                                  [0.071, 0.158, 0.284])

        train_transform = transforms.Compose([
            #transforms.Resize((384, 256)),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        strong_transform = transforms.Compose([
            #transforms.RandomResizedCrop(224),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            normalize,
        ])

        test_transform = transforms.Compose([
            #transforms.Resize((384, 256)),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = MuReD(root, "train", (train_transform,strong_transform))
        test_dataset = MuReD(root, "test", test_transform)
        #global_dataset = MuReD(root, "global", test_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = MuReD_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = MuReD_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = MuReD_noniid(train_dataset, args.num_users)


    elif args.dataset == "OIA":
        root = "../data/MuReD"
        #root = "../data/OIA-ODIR"
        normalize = transforms.Normalize([0.103, 0.190, 0.300],
                                         [0.108, 0.189, 0.285])

        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        strong_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = ODIRDataset(root, "train", (train_transform,strong_transform))
        test_dataset = ODIRDataset(root, "test", test_transform)
        #global_dataset = MuReD(root, "global", test_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = MuReD_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = MuReD_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = MuReD_noniid(train_dataset, args.num_users)



    return train_dataset, test_dataset, user_groups


def FedAvg_tao(t, weight, class_active_client_list = None):
    if class_active_client_list is None:
        t_avg = np.array([0.]*len(t[0]))
        for i, tao in enumerate(t):
            t_avg += tao * float(weight[i])
        t_avg = t_avg / float(sum(weight))
        return t_avg
    else:
        t_avg = np.array([0.] * len(t[0]))
        for cls, cls_active_clients in enumerate(class_active_client_list):
            weight_sum = 0.
            for i, tao in enumerate(t):
                if i in cls_active_clients:
                    t_avg[cls] += tao[cls] * float(weight[i])
                    weight_sum += float(weight[i])
            if len(cls_active_clients) == 0:
                t_avg[cls] = 1.
            else:
                t_avg[cls] = t_avg[cls] / weight_sum
        return t_avg


def FedAvg_proto(Prototypes, weight, class_active_client_list):
    Prototype_avg = torch.zeros((len(Prototypes[0]), len(Prototypes[0][0])))
    # Prototype_avg = np.array([torch.zeros_like(Prototypes[0][0])] * len(Prototypes[0]))
    for cls, cls_active_clients in enumerate(class_active_client_list):
        Prototype_class_0_avg = torch.zeros_like(Prototypes[0][0])
        Prototype_class_1_avg = torch.zeros_like(Prototypes[0][0])
        for client_id in cls_active_clients:
            Prototype_class_0_avg = Prototypes[client_id][2*cls] * weight[client_id] + Prototype_class_0_avg
            Prototype_class_1_avg = Prototypes[client_id][2*cls+1] * weight[client_id] + Prototype_class_1_avg
        # print(cls)
        # print(cls_active_clients)
        # print(Prototype_class_0_avg)
        # print(Prototype_class_1_avg)
        Prototype_class_0_avg = Prototype_class_0_avg / np.sum(np.array(weight)[cls_active_clients])
        Prototype_class_1_avg = Prototype_class_1_avg / np.sum(np.array(weight)[cls_active_clients])
        # print(Prototype_class_0_avg)
        # print(Prototype_class_1_avg)
        Prototype_avg[2*cls] = Prototype_class_0_avg
        Prototype_avg[2*cls+1] = Prototype_class_1_avg
        # print(Prototype_avg)
        # input()
    return Prototype_avg


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def aggregate_gmm_params(gmm_params_list, weights):
    # gmm_params_list 是所有客户端返回的 GMM 参数列表
    # weights 是客户端训练样本比例（用于加权）

    agg_params = {
        'weights': np.zeros(2),
        'means': np.zeros((2, 1)),
        'covariances': np.zeros((2, 1, 1))
    }

    for gmm, w in zip(gmm_params_list, weights):
        agg_params['weights'] += gmm['weights'] * w
        agg_params['means'] += gmm['means'] * w
        agg_params['covariances'] += gmm['covariances'] * w

    return agg_params



def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


class MuReD(Dataset):
    def  __init__(self, datapath, mode, transform=None):
        self.datapath = datapath
        self.mode = mode
        self.transform = transform

        assert self.mode in ["train", "test"]
        csv_file = os.path.join(self.datapath, self.mode + "_data.csv")
        self.file = pd.read_csv(csv_file)

        self.image_list = self.file["Image Index"].values
        self.targets = self.file.iloc[0:, 1:].values.astype(np.float32)

    def __getitem__(self, index: int):
        image_id, target = self.image_list[index], self.targets[index]
        image = self.read_image(image_id)
        #image = self.read_image(image_id+'.jpg')

        if self.transform is not None:
            if isinstance(self.transform, tuple):
                image1 = self.transform[0](image)
                image2 = self.transform[1](image)

                return {"image_aug_1": image1,
                        "image_aug_2": image2,
                        "target": target,
                        "index": index,
                        "image_id": image_id}
            else:
                image = self.transform(image)
                return {"image": image,
                        "target": target,
                        "index": index,
                        "image_id": image_id}

    def __len__(self):
        return len(self.targets)

    def read_image(self, image_id):
        #image_path = os.path.join(r"../data/MuReD/images", image_id)
        # image_path = os.path.join("../data/46/images", image_id)
        image_path = os.path.join("../data/46/images", image_id)

        image = Image.open(image_path).convert("RGB")
        return image


class ODIRDataset(Dataset):
    def __init__(self, images_path, df, transform=None):
        self.images_path = images_path
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        left_image_path = f'{self.images_path}/{self.df.at[index, "Left-Fundus"]}'
        right_image_path = f'{self.images_path}/{self.df.at[index, "Right-Fundus"]}'
        image_id = [self.df.at[index, "Left-Fundus"],self.df.at[index, "Right-Fundus"]]
        left_img = Image.open(left_image_path)
        right_img = Image.open(right_image_path)
        target = self.df.loc[index, ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']]
        target = target.to_numpy(dtype=np.float32)
        target = torch.tensor(target)
        if self.transform:
            if isinstance(self.transform, tuple):
                left_img = self.transform[0](left_img)
                right_img = self.transform[0](right_img)

                # left_img1 = self.transform[1](left_img)
                # right_img1 = self.transform[1](right_img)

                return {"image_aug_1": [left_img,right_img],
                        "image_aug_2": [left_img,right_img],
                        "target": target,
                        "index": index,
                        "image_id": image_id}
            else:
                left_img = self.transform(left_img)
                right_img = self.transform(right_img)
                return {"image": [left_img,right_img],
                        "target": target,
                        "index": index,
                        "image_id": image_id}


