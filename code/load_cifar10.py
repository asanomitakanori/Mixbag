import numpy as np
import pickle
import sys


def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()

    return data


def load_cifar10(dataset_dir='../dataset/'):
    X_train = None
    y_train = []

    for i in range(1, 6):
        data_dic = unpickle(
            dataset_dir+"cifar-10-batches-py/data_batch_{}".format(i))
        if i == 1:
            X_train = data_dic['data']
        else:
            X_train = np.vstack((X_train, data_dic['data']))
        y_train += data_dic['labels']

    test_data_dic = unpickle(dataset_dir+"cifar-10-batches-py/test_batch")
    X_test = test_data_dic['data']
    X_test = X_test.reshape(len(X_test), 3, 32, 32)
    y_test = np.array(test_data_dic['labels'])
    X_train = X_train.reshape((len(X_train), 3, 32, 32))
    y_train = np.array(y_train)

    train_img = X_train.transpose((0, 2, 3, 1))
    train_label = y_train
    test_img = X_test.transpose((0, 2, 3, 1))
    test_label = y_test

    return train_img, train_label, test_img, test_label


if __name__ == '__main__':
    train_img, train_label, test_img, test_label = load_cifar10()
    print('==========================')
    print('Dataset information')
    print('train_img.shape, train_label.shape: ')
    print(train_img.shape, train_label.shape)
    print('test_img.shape, test_label.shape: ')
    print(test_img.shape, test_label.shape)
    print('train_img.min(), train_img.max(): ')
    print(train_img.min(), train_img.max())
    print('train_label.min(), train_label.max(): ')
    print(train_label.min(), train_label.max())
    print('example: ')
    print(train_img[:5], train_label[:5])
    print('==========================')
