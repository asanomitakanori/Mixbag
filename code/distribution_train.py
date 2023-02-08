
import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn.functional as F
from time import time
from tqdm import tqdm
import logging
import random
from utils import fix_seed
import torchvision.transforms as transforms
from scipy.stats import norm
from hydra.utils import to_absolute_path as to_abs_path 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm

def train_net(args, model, optimizer, train_loader, loss_function_train, loss_function_val):
    fix_seed(args.seed)
    fig, ax = plt.subplots()
    # plt.rcParams["font.size"] = 30
    for epoch in range(args.num_epochs):
        c_a = ["r", "g", "b", "c", "m", "y", "k", "slategray", 'lightpink', 'crimson', 'goldenrod']
        ############ train ###################
        s_time = time()
        model.train()
        losses = []
        gt, pred = [], []
        for iteration, (data, label, lp, min_point, max_point, lp_list) in enumerate(tqdm(train_loader, leave=False)):
            if args.tsne == False:
                if iteration <= 8:
                    gt.append(lp-lp_list)
                    pred.append(max_point - min_point)
                else:
                    gt = np.concatenate(gt)
                    pred = np.concatenate(pred)
                    ax.grid(True)
                    ax.scatter(abs(gt), pred, c = pred, vmin=-10, vmax=6, cmap='Blues')
                    ax.set_xlabel('Difference between estimated proportions and correct proportions')
                    ax.set_ylabel('Confidential interval')
                    xticklabels = ax.get_xticklabels()
                    yticklabels = ax.get_yticklabels()
                    ax.set_axisbelow(True)
                    ax.set_ylim(-0.05,1)
                    ax.set_xticklabels(xticklabels,fontsize=12, rotation=0)
                    ax.set_yticklabels(yticklabels,fontsize=12)
                    plt.show()
                    print('a')
            else:
                if iteration <= 10:
                    gt.append(lp)
                    pred.append(lp_list)
                else:
                    gt = np.concatenate(gt)
                    pred = np.concatenate(pred)
                    x_reduced = TSNE(n_components=2, random_state=0).fit_transform(gt)
                    label2color = {0: 'tomato', 1:'dodgerblue'}
                    la = [label2color[p] for p in pred]
                    plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=la)
                    # ax.set_xlabel('Difference between estimated proportions and correct proportions')
                    # ax.set_ylabel('Confidential interval')
                    xticklabels = ax.get_xticklabels()
                    yticklabels = ax.get_yticklabels()
                    ax.grid(True)
                    ax.set_axisbelow(True)
                    ax.legend()
                    ax.set_xticklabels(xticklabels,fontsize=12, rotation=0)
                    ax.set_yticklabels(yticklabels,fontsize=12)
                    plt.show()
                    print('a')
    return train_loss, val_loss


def load_data_bags(args):  # Toy
    ######### load data #######
    test_data = np.load(to_abs_path('data/%s/test_data.npy' % (args.dataset)))
    test_label = np.load(to_abs_path('data/%s/test_label.npy' % (args.dataset)))

    train_bags = np.load(to_abs_path('data/%s/%d/train_bags.npy' % (args.dataset, args.fold)))
    train_labels = np.load(to_abs_path('data/%s/%d/train_labels.npy' % (args.dataset, args.fold)))
    train_lps = np.load(to_abs_path('data/%s/%d/train_original_lps.npy' % (args.dataset, args.fold)))  
    val_bags = np.load(to_abs_path('data/%s/%d/val_bags.npy' % (args.dataset, args.fold)))
    val_labels = np.load(to_abs_path('data/%s/%d/val_labels.npy' % (args.dataset, args.fold)))
    val_lps = np.load(to_abs_path('data/%s/%d/val_original_lps.npy' % (args.dataset, args.fold)))

    if args.tsne == False:
        train_dataset = DatasetBag_randomchoice_statistic(
                                                        args=args, 
                                                        data=train_bags, 
                                                        label=train_labels, 
                                                        lp=train_lps, 
                                                        )
    if args.tsne == True:
        train_dataset = Dataset_bag_TSNE(
                                        args=args, 
                                        data=train_bags, 
                                        label=train_labels, 
                                        lp=train_lps, 
                                        )

    train_loader = torch.utils.data.DataLoader(
                                                train_dataset, 
                                                batch_size=args.mini_batch, 
                                                shuffle=True,  
                                                num_workers=args.num_workers
                                                )



    return train_loader



class DatasetBag_randomchoice_statistic(torch.utils.data.Dataset):
    def __init__(self, args, data, label, lp):
        fix_seed(args.seed)
        self.standard_normal_value = args.standard_normal_value
        self.data = data
        self.label = label
        self.lp = lp
        self.classes = args.classes

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071), (0.2673))])
        self.len = len(self.data)
        self.choice = args.choice

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        choice_bags_num = np.random.randint(0, self.len)
        choice_bags = self.data[choice_bags_num]
        choice_labels = self.label[choice_bags_num]
        choice_lp = self.lp[choice_bags_num]
        data = self.data[idx]
        label = self.label[idx]
        lp = self.lp[idx]
        index_order1 =np.arange(data.shape[0])
        random.shuffle(index_order1)
        index_order2 =np.arange(data.shape[0])
        random.shuffle(index_order2)

        if self.choice == 'half':
            index_order1 = index_order1[0:index_order1.shape[0] // 2]
            index_order2 = index_order2[0:index_order2.shape[0] // 2]
            min_error, max_error = error_cover_area(lp, choice_lp, len(index_order1), len(index_order2), self.standard_normal_value)
            a: float = len(index_order1) / (len(index_order1) + len(index_order2))
            b: float =  len(index_order2) / (len(index_order1) + len(index_order2))
            lp: float = a*lp + b*choice_lp
            lp = torch.tensor(lp).float()

        elif self.choice == 'uniform':
            x = np.random.randint(1, data.shape[0])
            index_order1 = index_order1[0:x]
            index_order2 = index_order2[x:]
            min_error, max_error = error_cover_area(lp, choice_lp, len(index_order1), len(index_order2), self.standard_normal_value)
            a: float = len(index_order1) / (len(index_order1) + len(index_order2))
            b: float =  len(index_order2) / (len(index_order1) + len(index_order2))
            lp: float = a*lp + b*choice_lp
            lp = torch.tensor(lp).float()

        elif self.choice == 'gauss':
            x = np.random.normal(loc=0.5, scale=0.1, size=1)
            x = int(x * data.shape[0])
            if x == 0:
                x = 1
            elif x >= data.shape[0]:
                x = data.shape[-1]
            index_order1 = index_order1[0:x]
            index_order2 = index_order2[x:]
            min_error, max_error = error_cover_area(lp, choice_lp, len(index_order1), len(index_order2), self.standard_normal_value)
            a: float = len(index_order1) / (len(index_order1) + len(index_order2))
            b: float =  len(index_order2) / (len(index_order1) + len(index_order2))
            lp: float = a*lp + b*choice_lp
            lp = torch.tensor(lp).float()

        data = data[index_order1]
        label = label[index_order1]
        choice_bags = choice_bags[index_order2]
        choice_labels = choice_labels[index_order2]
        data = np.concatenate([data, choice_bags], axis=0)
        label = np.concatenate([label, choice_labels])
        label = torch.tensor(label).long()
        min_error, max_error = torch.tensor(min_error).float(), torch.tensor(max_error).float()

        if len(data.shape) == 3:
            data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
            (b, w, h, c) = data.shape
            trans_data = torch.zeros((b, c, w, h))
            for i in range(b):
                trans_data[i] = self.transform2(data[i])
        else:
            (b, w, h, c) = data.shape
            trans_data = torch.zeros((b, c, w, h))
            for i in range(b):
                trans_data[i] = self.transform(data[i])
        data = trans_data
        lp_list = [label[label==i].shape[0] for i in range(self.classes)]
        lp_list = np.array(lp_list)
        lp_list = lp_list / label.shape[0]
        return data, label, lp, min_error, max_error, lp_list


class Dataset_bag_TSNE(torch.utils.data.Dataset):
    def __init__(self, args, data, label, lp):
        fix_seed(args.seed)
        self.standard_normal_value = args.standard_normal_value
        self.data = data
        self.label = label
        self.lp = lp
        self.classes = args.classes

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071), (0.2673))])
        self.len = len(self.data)
        self.choice = args.choice

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        n = np.random.rand()
        if n >= 0.5:
            choice_bags_num = np.random.randint(0, self.len)
            choice_bags = self.data[choice_bags_num]
            choice_labels = self.label[choice_bags_num]
            choice_lp = self.lp[choice_bags_num]
            data = self.data[idx]
            label = self.label[idx]
            lp = self.lp[idx]
            index_order1 =np.arange(data.shape[0])
            random.shuffle(index_order1)
            index_order2 =np.arange(data.shape[0])
            random.shuffle(index_order2)

            if self.choice == 'half':
                index_order1 = index_order1[0:index_order1.shape[0] // 2]
                index_order2 = index_order2[0:index_order2.shape[0] // 2]
                min_error, max_error = error_cover_area(lp, choice_lp, len(index_order1), len(index_order2), self.standard_normal_value)
                a: float = len(index_order1) / (len(index_order1) + len(index_order2))
                b: float =  len(index_order2) / (len(index_order1) + len(index_order2))
                lp: float = a*lp + b*choice_lp
                lp = torch.tensor(lp).float()

            elif self.choice == 'uniform':
                x = np.random.randint(1, data.shape[0])
                index_order1 = index_order1[0:x]
                index_order2 = index_order2[x:]
                min_error, max_error = error_cover_area(lp, choice_lp, len(index_order1), len(index_order2), self.standard_normal_value)
                a: float = len(index_order1) / (len(index_order1) + len(index_order2))
                b: float =  len(index_order2) / (len(index_order1) + len(index_order2))
                lp: float = a*lp + b*choice_lp
                lp = torch.tensor(lp).float()

            elif self.choice == 'gauss':
                x = np.random.normal(loc=0.5, scale=0.1, size=1)
                x = int(x * data.shape[0])
                if x == 0:
                    x = 1
                elif x >= data.shape[0]:
                    x = data.shape[-1]
                index_order1 = index_order1[0:x]
                index_order2 = index_order2[x:]
                min_error, max_error = error_cover_area(lp, choice_lp, len(index_order1), len(index_order2), self.standard_normal_value)
                a: float = len(index_order1) / (len(index_order1) + len(index_order2))
                b: float =  len(index_order2) / (len(index_order1) + len(index_order2))
                lp: float = a*lp + b*choice_lp
                lp = torch.tensor(lp).float()

            data = data[index_order1]
            label = label[index_order1]
            choice_bags = choice_bags[index_order2]
            choice_labels = choice_labels[index_order2]
            data = np.concatenate([data, choice_bags], axis=0)
            label = np.concatenate([label, choice_labels])
            label = torch.tensor(label).long()
            min_error, max_error = torch.tensor(min_error).float(), torch.tensor(max_error).float()

            if len(data.shape) == 3:
                data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
                (b, w, h, c) = data.shape
                trans_data = torch.zeros((b, c, w, h))
                for i in range(b):
                    trans_data[i] = self.transform2(data[i])
            else:
                (b, w, h, c) = data.shape
                trans_data = torch.zeros((b, c, w, h))
                for i in range(b):
                    trans_data[i] = self.transform(data[i])
            data = trans_data
            return data, label, lp, 0, 0, 0

        else:
            min_error = 0
            max_error = 0
            data = self.data[idx]
            if len(data.shape) == 3:
                data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
                (b, w, h, c) = data.shape
                trans_data = torch.zeros((b, c, w, h))
                for i in range(b):
                    trans_data[i] = self.transform2(data[i])
            else:
                (b, w, h, c) = data.shape
                trans_data = torch.zeros((b, c, w, h))
                for i in range(b):
                    trans_data[i] = self.transform(data[i])
            data = trans_data

            label = self.label[idx]
            label = torch.tensor(label).long()
            min_error, max_error = torch.full((1, self.classes), -1).reshape(self.classes).float(), torch.full((1, self.classes), -1).reshape(self.classes).float()
            lp = self.lp[idx]
            lp = torch.tensor(lp).float()
            return data, label, lp, 1, 1, 1


def error_cover_area(
                    proportion1: list, 
                    proportion2: list,
                    sampling_num1: int, 
                    sampling_num2: int,               
                    confidence_interval: float
                    ):
    a: float = sampling_num1 / (sampling_num1 + sampling_num2)
    b: float = sampling_num2 / (sampling_num1 + sampling_num2)
    t = norm.isf(q = confidence_interval)
    min1 = proportion1 - t * np.sqrt(proportion1 * (1 - proportion1) / sampling_num1)
    min2 = proportion2 - t * np.sqrt(proportion2 * (1 - proportion2) / sampling_num2)
    max1 = proportion1 + t * np.sqrt((proportion1 * (1 - proportion1)) / sampling_num1)
    max2 = proportion2 + t * np.sqrt((proportion2 * (1 - proportion2)) / sampling_num2)
    plp_of_merging =  a * proportion1 +  b * proportion2
    confidence_area = t * (a * np.sqrt(proportion1*(1-proportion1) / sampling_num1) + b * np.sqrt(proportion2*(1-proportion2) / sampling_num2))
    min = plp_of_merging - confidence_area
    max = plp_of_merging + confidence_area
    return min, max

