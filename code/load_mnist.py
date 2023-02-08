import gzip
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms


def load_img(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28, 28)

    return data


def load_label(file_path):
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return labels


def load_mnist(dataset_dir='../dataset/MNIST/raw/', is_to_rgb=True):
    key_file = {
        'train_img': '/train-images-idx3-ubyte.gz',
        'train_label': '/train-labels-idx1-ubyte.gz',
        'test_img': '/t10k-images-idx3-ubyte.gz',
        'test_label': '/t10k-labels-idx1-ubyte.gz'
    }

    train_img = load_img(dataset_dir+key_file['train_img'])
    train_label = load_label(dataset_dir+key_file['train_label'])
    test_img = load_img(dataset_dir+key_file['test_img'])
    test_label = load_label(dataset_dir+key_file['test_label'])

    if is_to_rgb == True:
        train_img_rgb = [np.array(Image.fromarray(img).convert('RGB'))
                         for img in train_img]
        test_img_rgb = [np.array(Image.fromarray(img).convert('RGB'))
                        for img in test_img]
        train_img = np.array(train_img_rgb)
        test_img = np.array(test_img_rgb)

    return train_img, train_label, test_img, test_label


if __name__ == '__main__':
    train_img, train_label, test_img, test_label = load_mnist(
        dataset_dir='../dataset/MNIST/raw/',
        is_to_rgb=True)
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
