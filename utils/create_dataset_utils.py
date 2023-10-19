import gzip
import numpy as np
from PIL import Image

import pickle
import sys

import scipy.io as sio


def load_svhn(dataset_dir="../dataset/"):
    train_data = sio.loadmat(dataset_dir + "svhn/train_32x32.mat")
    x_train = train_data["X"]
    x_train = x_train.transpose((3, 0, 1, 2))
    y_train = train_data["y"].reshape(-1)
    y_train[y_train == 10] = 0

    test_data = sio.loadmat(dataset_dir + "svhn/test_32x32.mat")
    x_test = test_data["X"]
    x_test = x_test.transpose((3, 0, 1, 2))
    y_test = test_data["y"].reshape(-1)
    y_test[y_test == 10] = 0

    return x_train, y_train, x_test, y_test


def load_img(file_path):
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28, 28)

    return data


def load_label(file_path):
    with gzip.open(file_path, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return labels


def load_mnist(dataset_dir="../dataset/MNIST/raw/", is_to_rgb=True):
    key_file = {
        "train_img": "/train-images-idx3-ubyte.gz",
        "train_label": "/train-labels-idx1-ubyte.gz",
        "test_img": "/t10k-images-idx3-ubyte.gz",
        "test_label": "/t10k-labels-idx1-ubyte.gz",
    }

    train_img = load_img(dataset_dir + key_file["train_img"])
    train_label = load_label(dataset_dir + key_file["train_label"])
    test_img = load_img(dataset_dir + key_file["test_img"])
    test_label = load_label(dataset_dir + key_file["test_label"])

    if is_to_rgb == True:
        train_img_rgb = [
            np.array(Image.fromarray(img).convert("RGB")) for img in train_img
        ]
        test_img_rgb = [
            np.array(Image.fromarray(img).convert("RGB")) for img in test_img
        ]
        train_img = np.array(train_img_rgb)
        test_img = np.array(test_img_rgb)

    return train_img, train_label, test_img, test_label


def unpickle(file):
    fp = open(file, "rb")
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding="latin-1")
    fp.close()

    return data


def load_cifar10(dataset_dir="../dataset/"):
    X_train = None
    y_train = []

    for i in range(1, 6):
        data_dic = unpickle(dataset_dir + "cifar-10-batches-py/data_batch_{}".format(i))
        if i == 1:
            X_train = data_dic["data"]
        else:
            X_train = np.vstack((X_train, data_dic["data"]))
        y_train += data_dic["labels"]

    test_data_dic = unpickle(dataset_dir + "cifar-10-batches-py/test_batch")
    X_test = test_data_dic["data"]
    X_test = X_test.reshape(len(X_test), 3, 32, 32)
    y_test = np.array(test_data_dic["labels"])
    X_train = X_train.reshape((len(X_train), 3, 32, 32))
    y_train = np.array(y_train)

    train_img = X_train.transpose((0, 2, 3, 1))
    train_label = y_train
    test_img = X_test.transpose((0, 2, 3, 1))
    test_label = y_test

    return train_img, train_label, test_img, test_label
