import argparse
import os
import random
from glob import glob

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from utils.create_dataset_utils import load_cifar10, load_mnist, load_svhn


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_label_proportion(num_bags=100, num_classes=10):
    proportion = np.random.rand(num_bags, num_classes)
    proportion /= proportion.sum(axis=1, keepdims=True)
    return proportion


def get_N_label_proportion(proportion, num_instances, num_classes):
    N = np.zeros(proportion.shape)
    for i in range(len(proportion)):
        p = proportion[i]
        for c in range(len(p)):
            if (c + 1) != num_classes:
                num_c = int(np.round(num_instances * p[c]))
                if sum(N[i]) + num_c >= num_instances:
                    num_c = int(num_instances - sum(N[i]))
            else:
                num_c = int(num_instances - sum(N[i]))

            N[i][c] = int(num_c)
        np.random.shuffle(N[i])
    print(N.sum(axis=0))
    print((N.sum(axis=1) != num_instances).sum())
    return N


def create_bags_val(data, label, num_bags, num_instances, args):
    # make poroportion
    proportion = get_label_proportion(num_bags, args.num_classes)
    proportion_N = get_N_label_proportion(proportion, num_instances, args.num_classes)

    proportion_N_nega = np.zeros((0, args.num_classes))
    proportion_N_nega[:, 0] = num_instances

    proportion_N = np.concatenate([proportion_N, proportion_N_nega], axis=0)

    # make index
    idx = np.arange(len(label))
    idx_c = []
    for c in range(args.num_classes):
        idx_c.append(idx[label[idx] == c])

    bags_idx = []
    for n in range(len(proportion_N)):
        bag_idx = []
        for c in range(args.num_classes):
            sample_c_index = idx_c[c][0 : int(proportion_N[n][c])]
            idx_c[c] = idx_c[c][int(proportion_N[n][c]) :]
            bag_idx.extend(sample_c_index)

        np.random.shuffle(bag_idx)
        bags_idx.append(bag_idx)

    bags_idx = np.array(bags_idx)
    # bags_index.shape => (num_bags, num_instances)
    bags, labels = data[bags_idx], label[bags_idx]

    original_lps = proportion_N / num_instances

    partial_lps = original_lps.copy()
    posi_nega = original_lps[:, 0] != 1
    partial_lps[posi_nega == 1, 0] = 0  # mask negative class
    partial_lps /= partial_lps.sum(axis=1, keepdims=True)  # normalize

    return bags, labels, original_lps, partial_lps


def create_bags(data, label, num_bags, num_instances, args):
    # make poroportion
    proportion = get_label_proportion(num_bags, args.num_classes)
    proportion_N = get_N_label_proportion(proportion, num_instances, args.num_classes)

    proportion_N_nega = np.zeros((0, args.num_classes))
    proportion_N_nega[:, 0] = num_instances

    proportion_N = np.concatenate([proportion_N, proportion_N_nega], axis=0)

    # make index
    idx = np.arange(len(label))
    idx_c = []
    if args.dataset == "cifar10" or args.dataset == "svhn":
        for c in range(args.num_classes):
            idx_c.append(idx[label[idx] == c])
    else:
        for c in range(args.num_classes):
            idx_c.append(idx[(label[idx] == c).squeeze()])
    for i in range(len(idx_c)):
        random.shuffle(idx_c[i])

    bags_idx = []
    for n in range(len(proportion_N)):
        bag_idx = []
        for c in range(args.num_classes):
            sample_c_index = idx_c[c][0 : int(proportion_N[n][c])]
            idx_c[c] = idx_c[c][int(proportion_N[n][c]) :]
            bag_idx.extend(sample_c_index)

        np.random.shuffle(bag_idx)
        bags_idx.append(bag_idx)

    bags_idx = np.array(bags_idx)
    bags, labels = data[bags_idx], label[bags_idx]

    original_lps = proportion_N / num_instances

    partial_lps = original_lps.copy()
    posi_nega = original_lps[:, 0] != 1
    partial_lps[posi_nega == 1, 0] = 0  # mask negative class
    partial_lps /= partial_lps.sum(axis=1, keepdims=True)  # normalize

    return bags, labels, original_lps, partial_lps


def cifar10svhn(args):
    # load dataset
    if args.dataset == "mnist":
        data, label, test_data, test_label = load_mnist()
    elif args.dataset == "svhn":
        data, label, test_data, test_label = load_svhn()
    elif args.dataset == "cifar10":
        data, label, test_data, test_label = load_cifar10()

    # k-fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(skf.split(data, label)):
        train_data, train_label = data[train_idx], label[train_idx]
        val_data, val_label = data[val_idx], label[val_idx]

        output_path = "data/%s/%d/" % (
            args.dataset,
            i,
        )
        make_folder(output_path)

        # train
        bags, labels, original_lps, _ = create_bags(
            train_data,
            train_label,
            args.train_num_bags,
            args.train_num_instances,
            args,
        )
        np.save("%s/train_bags" % (output_path), bags)
        np.save("%s/train_labels" % (output_path), labels)
        np.save("%s/train_lps" % (output_path), original_lps)
        # val
        bags, labels, original_lps, _ = create_bags_val(
            val_data,
            val_label,
            args.val_num_bags,
            args.val_num_instances,
            args,
        )
        np.save("%s/val_bags" % (output_path), bags)
        np.save("%s/val_labels" % (output_path), labels)
        np.save("%s/val_lps" % (output_path), original_lps)
        # np.save("%s/val_partial_lps" % (output_path), partial_lps)

    # test
    used_test_data, used_test_label = [], []
    for c in range(args.num_classes):
        used_test_data.extend(test_data[test_label == c])
        used_test_label.extend(test_label[test_label == c])
    test_data, test_label = np.array(used_test_data), np.array(used_test_label)
    np.save(
        "data/%s/test_data" % (args.dataset,),
        test_data,
    )
    np.save(
        "data/%s/test_label" % (args.dataset,),
        test_label,
    )


def medmnist(args):
    dataset_list = glob("medmnist/*.npz")
    for data in dataset_list:
        a = np.load(data)
        print(a)
        args.dataset = data.split("/")[-1].split(".npz")[0]
        data_med = np.concatenate([a["train_images"], a["val_images"]])
        label_med = np.concatenate([a["train_labels"], a["val_labels"]])
        test_data, test_label = a["test_images"], a["test_labels"].squeeze()
        args.num_classes = label_med.max() + 1

        kf = KFold(n_splits=5, shuffle=True)
        for i, (train_idx, val_idx) in enumerate(kf.split(data_med)):
            train_data, train_label = data_med[train_idx], label_med[train_idx]
            val_data, val_label = data_med[val_idx], label_med[val_idx]

            output_path = "data/%s/%d/" % (
                args.dataset,
                i,
            )
            make_folder(output_path)

            # train
            bags, labels, original_lps, _ = create_bags(
                train_data,
                train_label,
                args.train_num_bags,
                args.train_num_instances,
                args,
            )
            np.save("%s/train_bags" % (output_path), bags)
            np.save("%s/train_labels" % (output_path), labels)
            np.save("%s/train_lps" % (output_path), original_lps)

            # val
            bags, labels, original_lps, _ = create_bags(
                val_data,
                val_label,
                args.val_num_bags,
                args.val_num_instances,
                args,
            )
            np.save("%s/val_bags" % (output_path), bags)
            np.save("%s/val_labels" % (output_path), labels)
            np.save("%s/val_lps" % (output_path), original_lps)
        used_test_data, used_test_label = [], []
        for c in range(args.num_classes):
            used_test_data.extend(test_data[test_label == c])
            used_test_label.extend(test_label[test_label == c])
        test_data, test_label = np.array(used_test_data), np.array(used_test_label)
        np.save(
            "data/%s/test_data" % (args.dataset),
            test_data,
        )
        np.save(
            "data/%s/test_label" % (args.dataset),
            test_label,
        )


if __name__ == "__main__":
    bags = 512
    instance = 10

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--dataset", default="none", type=str)
    parser.add_argument("--num_classes", default=10, type=int)

    parser.add_argument("--train_num_bags", default=bags, type=int)
    parser.add_argument("--train_num_instances", default=instance, type=int)
    parser.add_argument("--val_num_bags", default=10, type=int)
    parser.add_argument("--val_num_instances", default=64, type=int)

    parser.add_argument("--aug_bags_num", default=1, type=int)
    args = parser.parse_args()

    #################

    np.random.seed(args.seed)
    args.dataset = "svhn"
    cifar10svhn(args)
    args.dataset = "cifar10"
    cifar10svhn(args)
    # medmnist
    medmnist(args)
