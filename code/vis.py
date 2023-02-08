import argparse
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from glob import glob
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from load_mnist import load_mnist
from utils import Dataset, DatasetBag, cal_mIoU, fix_seed, save_confusion_matrix, make_folder
from model import Attention


def vis_tsne(data, label, path):
    data2d = TSNE(n_components=2).fit_transform(data)
    # data2d = PCA(n_components=2).fit_transform(data)

    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i in range(label.max()+1):
        target = data2d[label == i]
        ax.scatter(x=target[:, 0], y=target[:, 1], label=i, alpha=0.5) 
           
    target = data2d[label == -1]
    ax.scatter(x=target[:, 0], y=target[:, 1], label='nega (0)', alpha=0.5, marker='x')

    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=14)
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.savefig(path)


def main(args):
    fix_seed(args.seed)


    make_folder(args.output_path)
    args.output_path += 'debug/'
    make_folder(args.output_path)
    # args.output_path += '%s/' % (args.dataset)
    # make_folder(args.output_path)
    # args.output_path += 'w_n_%s_w_p_%s_w_MIL_%s/' % (args.w_n, args.w_p, args.w_MIL)
    # make_folder(args.output_path)

    ######### load data #######
    test_data = np.load('data/%s/test_data.npy'%(args.dataset))
    test_label = np.load('data/%s/test_label.npy'%(args.dataset))
    bags = np.load('data/%s/bags.npy'%(args.dataset))
    labels = np.load('data/%s/labels.npy'%(args.dataset))
    lps = np.load('data/%s/lps.npy'%(args.dataset))

    train_bags, val_bags, train_labels, val_labels, train_lps, val_lps = train_test_split(
        bags, labels, lps, test_size=args.split_ratio, random_state=args.seed)

    train_dataset = DatasetBag(
        data=train_bags, label=train_labels, lp=train_lps)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.mini_batch,
        shuffle=False,  num_workers=args.num_workers)

    test_dataset = Dataset(data=test_data, label=test_label)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False,  num_workers=args.num_workers)

    fix_seed(args.seed)
    model = Attention(args.w_n, args.w_MIL)
    model = model.to(args.device)

    model_path = glob(args.output_path+'*.pkl')[0]
    model.load_state_dict(torch.load(model_path))

    model.eval()


    ############ train ###################
    train_f, train_gt = [], []
    with torch.no_grad():
        for data, label, _ in tqdm(train_loader, leave=False):
            bag_label = ((label!=0).sum(-1)!=0)
            for i in range(len(bag_label)):
                if bag_label[i]==0:
                    label[i][label[i]==0] = -1

            data = data.to(args.device)
            (N, B, C, W, H) = data.size()
            data = data.reshape(-1, C, W, H)
            f = model.feature_extraction(data)

            train_f.extend(f.cpu().detach().numpy())
            train_gt.extend(label.reshape(-1).numpy())

    ################## test ###################
    test_f, test_gt = [], []
    with torch.no_grad():
        for data, label in tqdm(test_loader, leave=False):
            data = data.to(args.device)
            f = model.feature_extraction(data)

            test_f.extend(f.cpu().detach().numpy())
            test_gt.extend(label.reshape(-1).cpu().detach().numpy())

    train_f, train_gt = np.array(train_f), np.array(train_gt)
    test_f, test_gt = np.array(test_f), np.array(test_gt)

    vis_tsne(train_f, train_gt, args.output_path+'/train_feature.png')
    vis_tsne(test_f, test_gt, args.output_path+'/test_feature.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42,
                        type=int, help='seed value')
    parser.add_argument('--dataset', default='mnist',
                        type=str, help='name of dataset')
    parser.add_argument('--device', default='cuda:0',
                        type=str, help='device')
    parser.add_argument('--split_ratio', default=0.25,
                        type=float, help='split ratio')
    parser.add_argument('--model', default='transformer', type=str,
                        help='name of used model.')
    parser.add_argument('--batch_size', default=512,
                        type=int, help='batch size for training.')
    parser.add_argument('--mini_batch', default=4,
                        type=int, help='mini batch for training.')
    parser.add_argument('--num_epochs', default=1000, type=int,
                        help='number of epochs for training.')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='number of workers for training.')
    parser.add_argument('--patience', default=10,
                        type=str, help='patience of early stopping')
    parser.add_argument('--lr', default=3e-4,
                        type=float, help='learning rate')
    parser.add_argument('--w_n', default=1,
                        type=float, help='weight of negative proportion loss')
    parser.add_argument('--w_p', default=1,
                        type=float, help='weight of positive proportion loss')
    parser.add_argument('--w_MIL', default=1,
                        type=float, help='weight of MIL loss')
    parser.add_argument('--output_path',
                        default='result/', type=str, help="output file name")
    args = parser.parse_args()

    #################
    main(args)
