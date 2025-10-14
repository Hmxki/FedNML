import copy
import logging

import numpy as np


def add_noise(args, y_train, dict_users, max_1=None):
    np.random.seed(args.seed)
    gamma_s = np.array([1.] * args.num_users)
    np.random.shuffle(gamma_s)
    gamma_c = gamma_s * args.noise_r
    y_train_noisy = copy.deepcopy(y_train)
    noise_idx_dict = [[] for i in range(len(dict_users))]
    if args.n_type == "instance":
        # 计算每个类别的总体分布作为基础概率
        class_dist = np.mean(y_train, axis=0)
        class_dist = class_dist / np.sum(class_dist)  # 归一化

        real_noise_level = np.zeros(args.num_users)
        for i in np.where(gamma_c > 0)[0]:
            sample_idx = np.array(list(dict_users[i]))
            hard_label_this_client = y_train[sample_idx]

            # 为每个样本生成模拟的soft_label
            soft_label_this_client = np.zeros((len(sample_idx), args.num_classes))

            for j in range(len(sample_idx)):
                # 多标签处理：基于原始标签生成基础概率
                original_labels = np.where(hard_label_this_client[j] == 1)[0]
                soft_label = np.random.dirichlet(np.ones(args.num_classes) * 0.1)

                # 增强原始标签类的概率
                for label in original_labels:
                    soft_label[label] += 0.5 / len(original_labels)

                soft_label = soft_label / np.sum(soft_label)

                # 混合全局分布
                soft_label = 0.7 * soft_label + 0.3 * class_dist
                soft_label_this_client[j] = soft_label

            # 计算每个样本的误分类概率（基于原始标签的平均概率）
            p_t = np.array([
                np.mean(soft_label_this_client[j][np.where(hard_label_this_client[j] == 1)[0]])
                if np.any(hard_label_this_client[j] == 1)  # 只有存在正类标签时才计算
                else np.mean(soft_label_this_client[j])
                for j in range(len(sample_idx))
            ])
            p_f = 1 - p_t
            p_f = p_f / p_f.sum()  # 归一化

            # 选择要添加噪声的样本
            noisy_idx = np.random.choice(np.arange(len(sample_idx)),
                                         size=int(gamma_c[i] * len(sample_idx)),
                                         replace=False,
                                         p=p_f)

            # 添加噪声
            for j in noisy_idx:
                # 创建临时soft_label并减弱原始标签的概率
                temp_soft = soft_label_this_client[j].copy()
                original_labels = np.where(hard_label_this_client[j] == 1)[0]
                temp_soft[original_labels] *= 0.2  # 降低原始标签的概率
                temp_soft = temp_soft / np.sum(temp_soft)

                # 随机选择要翻转的标签数量(1到max_1-1个)
                num_flips = np.random.randint(1, max(2, min(max_1, args.num_classes)))
                new_labels = np.random.choice(
                    np.arange(args.num_classes),
                    size=num_flips,
                    replace=False,
                    p=temp_soft)

                # 创建新标签
                new_label = np.zeros(args.num_classes)
                new_label[new_labels] = 1
                y_train_noisy[sample_idx[j]] = new_label

            noise_ratio = np.mean(
                np.any(y_train[sample_idx] != y_train_noisy[sample_idx], axis=1))
            noise_idx_dict[i]=sample_idx[noisy_idx]
            print("Client %d, noise level: %.4f, real noise ratio: %.4f" % (
                i, gamma_c[i], noise_ratio))
            real_noise_level[i] = noise_ratio
    return (y_train_noisy, gamma_s, real_noise_level, noise_idx_dict)