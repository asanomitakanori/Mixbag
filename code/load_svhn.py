import numpy as np
import scipy.io as sio
from hydra.utils import to_absolute_path as abs_path


def load_svhn(dataset_dir='../dataset/'):
    train_data = sio.loadmat(abs_path(dataset_dir + 'svhn/train_32x32.mat'))
    x_train = train_data['X']
    x_train = x_train.transpose((3, 0, 1, 2))
    y_train = train_data['y'].reshape(-1)
    y_train[y_train == 10] = 0

    test_data = sio.loadmat(dataset_dir + 'svhn/test_32x32.mat')
    x_test = test_data['X']
    x_test = x_test.transpose((3, 0, 1, 2))
    y_test = test_data['y'].reshape(-1)
    y_test[y_test == 10] = 0

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    train_img, train_label, test_img, test_label = load_svhn()
    print('==========================')
    print('Dataset information')
    print('train_img.shape, train_label.shape: ')
    print(train_img.shape, train_label.shape)
    print('test_img.shape, test_label.shape: ')
    print(test_img.shape, test_label.shape)
    # print('train_img.min(), train_img.max(): ')
    # print(train_img.min(), train_img.max())
    # print('train_label.min(), train_label.max(): ')
    # print(train_label.min(), train_label.max())
    # print(train_label)
    # print('example: ')
    # print(train_img[:5], train_label[:5])
    print('==========================')
