import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from utils import make_folder
from load_svhn import load_svhn
from load_cifar10 import load_cifar10
from load_mnist import load_mnist
import random
from hydra.utils import to_absolute_path as to_abs_path

def get_label_proportion(num_bags=100, num_classes=10):
    proportion = np.random.rand(num_bags, num_classes)
    proportion /= proportion.sum(axis=1, keepdims=True)

    return proportion


def get_N_label_proportion(proportion, num_instances, num_classes):
    N = np.zeros(proportion.shape)
    for i in range(len(proportion)):
        p = proportion[i]
        for c in range(len(p)):
            if (c+1) != num_classes:
                num_c = int(np.round(num_instances*p[c]))
                if sum(N[i])+num_c >= num_instances:
                    num_c = int(num_instances-sum(N[i]))
            else:
                num_c = int(num_instances-sum(N[i]))

            N[i][c] = int(num_c)
        np.random.shuffle(N[i])
    print(N.sum(axis=0))
    print((N.sum(axis=1) != num_instances).sum())
    return N


def create_bags(data, label, num_posi_bags, num_nega_bags, args):
    # make poroportion
    proportion = get_label_proportion(num_posi_bags, args.num_classes)
    proportion_N = get_N_label_proportion(
        proportion, args.num_instances, args.num_classes)

    proportion_N_nega = np.zeros((num_nega_bags, args.num_classes))
    proportion_N_nega[:, 0] = args.num_instances

    proportion_N = np.concatenate([proportion_N, proportion_N_nega], axis=0)

    # make index
    idx = np.arange(len(label))
    idx_c = []
    for c in range(args.num_classes):
        idx_c.append(idx[label[idx] == c])
    random.shuffle(idx_c[0])
    random.shuffle(idx_c[1])
    random.shuffle(idx_c[2])
    idx_c[0] = idx_c[0][0:500]
    idx_c[1] = idx_c[1][0:500]
    idx_c[2] = idx_c[2][0:500]

    bags_idx = []
    for n in range(len(proportion_N)):
        bag_idx = []
        for c in range(args.num_classes):
            sample_c_index = np.random.choice(
                idx_c[c], size=int(proportion_N[n][c]), replace=False)
            bag_idx.extend(sample_c_index)

        np.random.shuffle(bag_idx)
        bags_idx.append(bag_idx)

    bags_idx = np.array(bags_idx)
    # bags_index.shape => (num_bags, num_instances)
    # make data, label, proportion
    bags, labels = data[bags_idx], label[bags_idx]
    bags_temp, labels_temp = bags.copy(), labels.copy()
    split_bags_tmp, split_labels_tmp = bags_temp[:, 0:bags_temp.shape[1]//2], labels_temp[:, 0:labels_temp.shape[1]//2]
    split_bags_tmp = np.concatenate([np.expand_dims(split_bags_tmp[-1], 0), split_bags_tmp[:-1]])
    split_labels_tmp = np.concatenate([np.expand_dims(split_labels_tmp[-1], 0), split_labels_tmp[:-1]])

    bags_new, labels_new = bags[:, bags.shape[1]//2:], labels_temp[:, labels_temp.shape[1]//2:]
    bags_new = np.concatenate([split_bags_tmp, bags_new], axis=1)
    labels_new = np.concatenate([split_labels_tmp, labels_new], axis=1)

    class0_plp, class1_plp, class2_plp = np.sum(np.where(labels_new==0, True, False), axis=1), np.sum(np.where(labels_new==1, True, False), axis=1), np.sum(np.where(labels_new==2, True, False), axis=1)
    new_Proportion = np.stack([class0_plp, class1_plp, class2_plp]).transpose(1, 0)
    proportion_N = np.concatenate([proportion_N, proportion_N], axis=0)
    bags = np.concatenate([bags,bags_new], axis=0)
    labels = np.concatenate([labels,labels_new], axis=0)

    original_lps = proportion_N / args.num_instances

    partial_lps = original_lps.copy()
    posi_nega = (original_lps[:, 0] != 1)
    partial_lps[posi_nega == 1, 0] = 0  # mask negative class
    partial_lps /= partial_lps.sum(axis=1, keepdims=True)  # normalize

    return bags, labels, original_lps, partial_lps


def main(args):
    # load dataset
    if args.dataset == 'mnist':
        data, label, test_data, test_label = load_mnist()
    elif args.dataset == 'svhn':
        data, label, test_data, test_label = load_svhn()
    elif args.dataset == 'cifar10':
        data, label, test_data, test_label = load_cifar10()

    # k-fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(skf.split(data, label)):
        train_data, train_label = data[train_idx], label[train_idx]
        val_data, val_label = data[val_idx], label[val_idx]

        output_path = 'data/%s-%dclass-ins%d-bags%d-change-wrongplp/%d/' % (
            args.dataset, args.num_classes, args.num_instances, args.train_num_posi_bags,i)
        make_folder(to_abs_path(output_path))

        # train
        bags, labels, original_lps, partial_lps = create_bags(train_data, train_label,
                                                              args.train_num_posi_bags,
                                                              args.train_num_nega_bags,
                                                              args)
        np.save('%s/train_bags' % (output_path), bags)
        np.save('%s/train_labels' % (output_path), labels)
        np.save('%s/train_original_lps' % (output_path), original_lps)
        # np.save('%s/train_partial_lps' % (output_path), partial_lps)

        # val
        bags, labels, original_lps, partial_lps = create_bags(val_data, val_label,
                                                              args.val_num_posi_bags,
                                                              args.val_num_nega_bags,
                                                              args)
        np.save('%s/val_bags' % (output_path), bags)
        np.save('%s/val_labels' % (output_path), labels)
        np.save('%s/val_original_lps' % (output_path), original_lps)
        # np.save('%s/val_partial_lps' % (output_path), partial_lps)

    # test

    used_test_data, used_test_label = [], []
    for c in range(args.num_classes):
        used_test_data.extend(test_data[test_label == c])
        used_test_label.extend(test_label[test_label == c])
    test_data, test_label = np.array(used_test_data), np.array(used_test_label)
    np.save('data/%s-%dclass-ins%d-bags%d-change-wrongplp/test_data' %
            (args.dataset, args.num_classes, args.num_instances, args.train_num_posi_bags), test_data)
    np.save('data/%s-%dclass-ins%d-bags%d-change-wrongplp/test_label' %
            (args.dataset, args.num_classes, args.num_instances, args.train_num_posi_bags), test_label)


# def CRC100K():
#     negative_class = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM']
#     positive_class = ['STR', 'TUM']

#     # negative_path_list, positive_path_list = [], []
#     data, label = [], []
#     for c in negative_class:
#         for p in tqdm(glob('../dataset/NCT-CRC-HE-100K/%s/*' % c)):
#             data.append(np.asarray(Image.open(p).convert('RGB')))
#             label.append(0)

#     for i, c in tqdm(enumerate(positive_class)):
#         for p in glob('../dataset/NCT-CRC-HE-100K/%s/*' % c):
#             data.append(np.asarray(Image.open(p).convert('RGB')))
#             label.append(i+1)

#     np.save('../dataset/NCT-CRC-HE-100K/data', np.array(data))
#     np.save('../dataset/NCT-CRC-HE-100K/label', np.array(label))


if __name__ == '__main__':

    for i in [80, 120, 160, 200, 240]:
        parser = argparse.ArgumentParser()

        parser.add_argument('--seed', default=42, type=int)

        parser.add_argument('--dataset', default='cifar10', type=str)
        parser.add_argument('--num_classes', default=3, type=int)
        parser.add_argument('--num_instances', default=32, type=int)

        parser.add_argument('--train_num_posi_bags', default=i, type=int)
        parser.add_argument('--train_num_nega_bags', default=0, type=int)
        parser.add_argument('--val_num_posi_bags', default=10, type=int)
        parser.add_argument('--val_num_nega_bags', default=0, type=int)

        args = parser.parse_args()

        #################
        np.random.seed(args.seed)
        main(args)
    # CRC100K()