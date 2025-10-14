#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--num_users', type=int, default=4,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=3,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=32,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='ResNet', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=3, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='MuReD', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=23, help="number \
                        of classes")
    parser.add_argument('--gpu_id', default=0, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # noise
    parser.add_argument('--noise_r', type=float, default=0.4, help='noise_rate')
    parser.add_argument('--n_type', type=str, default='instance', help='type of noise')

    # missing
    parser.add_argument('--annotation_num', type=int, default=19, help='missing classes')


    parser.add_argument('--temperature', type=float, default=0.1, help='temperature')
    parser.add_argument('--entropy_t', type=float, default=0.1, help='entropy_t')

    parser.add_argument('--exp', type=str, default='FedMLP', help='method')

    # fedmlp
    parser.add_argument('--rounds_FedMLP_stage1', type=int, default=100, help='rounds')
    parser.add_argument('--U', type=float, default=0.7, help='tao_upper_bound')
    parser.add_argument('--L', type=float, default=0.3, help='tao_lower_bound')
    parser.add_argument('--clean_threshold', type=float, default=0.005, help='clean_threshold')
    parser.add_argument('--noise_threshold', type=float, default=0.01, help='noise_threshold')

    # fedlsm
    parser.add_argument('--uncertain_pool_size', type=float, default=0.2, help='uncertain_samples')
    parser.add_argument('--confident_pool_size', type=float, default=0.2, help='confident_samples')
    parser.add_argument('--alpha', type=float, default=0.2, help='pseudo temperate')

    parser.add_argument('--pseudo_positive_thresh', type=float, default=0.85, help='pseudo temperate')
    parser.add_argument('--pseudo_negative_thresh', type=float, default=0.1, help='pseudo temperate')
    parser.add_argument('--uncertain_label_p_thresh', type=float, default=0.7, help='pseudo temperate')
    parser.add_argument('--uncertain_label_n_thresh', type=float, default=0.05, help='pseudo temperate')


    parser.add_argument('--mini_batch_size_distillation', type=int, default=128)
    parser.add_argument('--lamda', type=float, default=1)

    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--alpha1', type=float, default=1, help="0.1,1,5")
    parser.add_argument('--beta', type=float, default=5,
                        help="coefficient for local proximal, 0 for fedavg, 1 for fedprox, 5 for noise fl")
    parser.add_argument('--correction', type=bool,default=True, help="whether to correct noisy labels")
    parser.add_argument('--relabel_ratio', type=float, default=0.7,
                        help="proportion of relabeled samples among selected noisy samples")
    parser.add_argument('--confidence_thres', type=float, default=0.5,
                        help="threshold of model's confidence on each sample")
    parser.add_argument('--clean_set_thres', type=float, default=0.1,
                        help="threshold of estimated noise level to filter 'clean' set used in fine-tuning stage")




    args = parser.parse_args()
    return args


# 4  8  4