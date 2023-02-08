from load_svhn import load_svhn
from load_cifar10 import load_cifar10
from load_mnist import load_mnist
import random
import numpy as np
import os
from hydra.utils import to_absolute_path as abs_path

def get_label_proportion(num_bags=100, num_classes=10):
    np.random.seed(0)
    proportion = np.random.rand(num_bags, num_classes)
    proportion /= proportion.sum(axis=1, keepdims=True)
    return proportion

# def get_label_proportion(num_bags=100, num_classes=10):
#     np.random.seed(0)

#     np_proportion = np.random.rand(num_bags, 2)
#     np_proportion /= np_proportion.sum(axis=1, keepdims=True)

#     n_proportion = np_proportion[:, 0][:, np.newaxis]

#     p_proportion = np.random.rand(num_bags, num_classes-1)
#     p_proportion /= p_proportion.sum(axis=1, keepdims=True)
#     p_proportion = p_proportion * \
#         np_proportion[:, 1][:, np.newaxis].repeat(num_classes-1, axis=1)

#     label_proportion = np.concatenate([n_proportion, p_proportion], axis=1)

#     # create new label proportion

#     return label_proportion


if __name__ == '__main__':
    dataset = 'cifar10'
    num_classes = 3
    num_instances = 32
    num_posi_bags, num_nega_bags = 40, 0

    if dataset == 'mnist':
        train_data, train_label, test_data, test_label = load_mnist()
    elif dataset == 'svhn':
        train_data, train_label, test_data, test_label = load_svhn()
    elif dataset == 'cifar10':
        train_data, train_label, test_data, test_label = load_cifar10()
    
    x = f'data/{dataset}-ins{num_instances}-bags{num_posi_bags}'
    os.makedirs(abs_path(x)) if os.path.isdir(x) is False else None

    LP_posi = get_label_proportion(num_posi_bags, num_classes)
    lp_nega = np.zeros(num_classes)
    lp_nega[0] = 1
    LP_nega = np.tile(lp_nega, num_nega_bags).reshape(-1, num_classes)

    N_train = len(train_data)
    index = np.arange(N_train)
    np.random.seed(0)
    np.random.shuffle(index)

    used_index = []
    for c in range(num_classes):
        used_index.append(index[train_label[index] == c])

    bags, labels, lps = [], [], []

    cnt = np.zeros(num_classes).astype(int)
    for i in range(num_posi_bags):
        bag, label = [], []
        for c in range(num_classes):
            c_index = used_index[c]
            if (c+1) != num_classes:
                num_c = int(num_instances*LP_posi[i][c])
            else:
                num_c = int(num_instances-len(bag))

            bag.extend(train_data[used_index[c][cnt[c]: cnt[c]+num_c]])
            label.extend(train_label[used_index[c][cnt[c]: cnt[c]+num_c]])
            cnt[c] += num_c
        print(cnt)

        bags.append(bag)
        labels.append(label)

    bags, labels = np.array(bags), np.array(labels)

    np.save(f'{x}/bags', bags)
    np.save(f'{x}/labels', labels)

    proportion = []
    for i in range(labels.shape[0]):
        classes = []
        for j in range(num_classes):
            classes.append((labels[i] == j).sum())
        classes = np.array(classes)
        proportion.append(np.array(classes) / classes.sum())
    proportion = np.array(proportion)

    np.save(f'{x}/lps', proportion)
    np.save(f'{x}/origin_lps', proportion)

    print(bags.shape, labels.shape, proportion.shape)

    (_, _, w, h, c) = bags.shape
    np.save(f'{x}/train_data', bags.reshape(-1, w, h, c))
    np.save(f'{x}/train_label', labels.reshape(-1))

    used_test_data, used_test_label = [], []
    for c in range(num_classes):
        used_test_data.extend(test_data[test_label == c])
        used_test_label.extend(test_label[test_label == c])
    test_data, test_label = np.array(used_test_data), np.array(used_test_label)
    np.save(f'{x}/test_data', test_data)
    np.save(f'{x}/test_label', test_label)
