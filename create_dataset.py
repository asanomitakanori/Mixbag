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


def get_label_proportion(num_bags: int, num_classes: int):
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


def create_bags_val(data, label, num_bags, num_instances, dataset):
    # make poroportion
    proportion = get_label_proportion(num_bags, args.num_classes)
    proportion_N = get_N_label_proportion(proportion, num_instances, args.num_classes)

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

    lps = proportion_N / num_instances

    return bags, labels, lps


def create_bags(data, label, num_bags, num_instances, dataset):
    # make poroportion
    proportion = get_label_proportion(num_bags, args.num_classes)
    proportion_N = get_N_label_proportion(proportion, num_instances, args.num_classes)

    # make index
    idx = np.arange(len(label))
    idx_c = []
    if dataset == "cifar10" or dataset == "svhn":
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

    lps = proportion_N / num_instances

    return bags, labels, lps


class CreateData(object):
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset

    def cifar10svhn(self):
        # load dataset
        for dataset in ["cifar10", "svhn"]:
            if dataset == "mnist":
                data, label, test_data, test_label = load_mnist()
            elif dataset == "svhn":
                data, label, test_data, test_label = load_svhn()
            elif dataset == "cifar10":
                data, label, test_data, test_label = load_cifar10()
            self.save_data(dataset, data, label, test_data, test_label)

    def medmnist(self):
        dataset_list = glob("medmnist/*.npz")
        for data in dataset_list:
            a = np.load(data)
            print(a)
            dataset = data.split("/")[-1].split(".npz")[0]
            data = np.concatenate([a["train_images"], a["val_images"]])
            label = np.concatenate([a["train_labels"], a["val_labels"]])
            test_data, test_label = (
                a["test_images"],
                a["test_labels"].squeeze(),
            )
            self.args.num_classes = label.max() + 1

            self.save_data(dataset, data, label, test_data, test_label)

    def save_data(self, dataset, data, label, test_data, test_label):
        if dataset in ["cifar10", "svhn"]:
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            all = kf.split(data, label)
        else:
            kf = KFold(n_splits=5, shuffle=True)
            all = kf.split(data)

        for i, (train_idx, val_idx) in enumerate(all):
            train_data, train_label = data[train_idx], label[train_idx]
            val_data, val_label = data[val_idx], label[val_idx]

            output_path = "data/%s/%d/" % (
                dataset,
                i,
            )
            make_folder(output_path)

            # train
            bags, labels, original_lps = create_bags(
                train_data,
                train_label,
                self.args.train_num_bags,
                self.args.train_num_instances,
                dataset,
            )
            np.save(f"{output_path}/train_bags", bags)
            np.save(f"{output_path}/train_labels", labels)
            np.save(f"{output_path}/train_lps", original_lps)

            # val
            bags, labels, original_lps = create_bags(
                val_data,
                val_label,
                self.args.val_num_bags,
                self.args.val_num_instances,
                dataset,
            )
            np.save(f"{output_path}/val_bags", bags)
            np.save(f"{output_path}/val_labels", labels)
            np.save(f"{output_path}/val_lps", original_lps)

        used_test_data, used_test_label = [], []
        for c in range(self.args.num_classes):
            used_test_data.extend(test_data[test_label == c])
            used_test_label.extend(test_label[test_label == c])
        test_data, test_label = np.array(used_test_data), np.array(used_test_label)
        np.save(
            f"data/{dataset}/test_data",
            test_data,
        )
        np.save(
            f"data/{dataset}/test_label",
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

    args = parser.parse_args()

    #################

    np.random.seed(args.seed)

    run = CreateData(args)
    run.cifar10svhn()
    run.medmnist()
