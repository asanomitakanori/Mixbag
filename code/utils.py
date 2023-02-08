import os
from sqlalchemy import false
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random
from losses import cross_entropy_loss
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = self.data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(data)
        label = self.label[idx]
        label = torch.tensor(label).long()
        return data, label


class DatasetBag(torch.utils.data.Dataset):
    def __init__(self, args, data, label, lp, augment=True):
        np.random.seed(args.seed)
        self.data = data
        self.label = label
        self.lp = lp
        self.augment = augment

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        (b, w, h, c) = data.shape
        trans_data = torch.zeros((b, c, w, h))
        for i in range(b):
            trans_data[i] = self.transform(data[i])
        data = trans_data

        label = self.label[idx]
        label = torch.tensor(label).long()

        lp = self.lp[idx]
        lp = torch.tensor(lp).float()

        return data, label, lp
        

class DatasetBag_WSI(torch.utils.data.Dataset):
    def __init__(self, name, path, label, proportion):
        self.name_list = name
        self.path_list = path
        self.label_list = label
        self.proportion_list = proportion
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = len(self.name_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        name = self.name_list[idx]

        data_list = []
        for path in self.path_list[name]:
            data = Image.open(path)
            data = np.asarray(data.convert('RGB'))
            data_list.append(data)
        data_list = np.array(data_list)

        (b, w, h, c) = data_list.shape
        data = torch.zeros((b, c, w, h))
        for i in range(b):
            data[i] = self.transform(data_list[i])

        label = self.label_list[name]
        label = torch.tensor(label).long()

        proportion = self.proportion_list[name]
        proportion = torch.tensor(proportion).float()

        return data, label, proportion


class DatasetBagSampling(torch.utils.data.Dataset):
    def __init__(self, name, path, label, proportion, num_sampled_instances):
        self.name_list = name
        self.path_list = path
        self.label_list = label
        self.proportion_list = proportion
        self.num_sampled_instances = num_sampled_instances
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = len(self.name_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        name = self.name_list[idx]

        data_list = []
        for path in self.path_list[name]:
            data = Image.open(path)
            data = np.asarray(data.convert('RGB'))
            data_list.append(data)
        data_list = np.array(data_list)

        (b, w, h, c) = data_list.shape
        if b > self.num_sampled_instances:
            index = np.arange(b)
            sampled_index = np.random.choice(index, self.num_sampled_instances)

            data = torch.zeros((self.num_sampled_instances, c, w, h))
            label = torch.zeros((self.num_sampled_instances))
            p_label = torch.zeros((self.num_sampled_instances))
            for i, j in enumerate(sampled_index):
                data[i] = self.transform(data_list[j])
                label[i] = self.label_list[name][j]
        else:
            data = torch.zeros((b, c, w, h))
            label = torch.zeros((b))
            p_label = torch.zeros((b))
            for i in range(b):
                data[i] = self.transform(data_list[i])
                label[i] = self.label_list[name][i]

        label = label.long()
        proportion = self.proportion_list[name]
        proportion = torch.tensor(proportion).float()

        return data, label, proportion


def save_confusion_matrix(cm, path, title=''):
    sns.heatmap(cm, annot=True, cmap='Blues_r', fmt='.4f')
    plt.xlabel('pred')
    plt.ylabel('GT')
    plt.title(title)
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def cal_OP_PC_mIoU(cm):
    num_classes = cm.shape[0]

    TP_c = np.zeros(num_classes)
    for i in range(num_classes):
        TP_c[i] = cm[i][i]

    FP_c = np.zeros(num_classes)
    for i in range(num_classes):
        FP_c[i] = cm[i, :].sum()-cm[i][i]

    FN_c = np.zeros(num_classes)
    for i in range(num_classes):
        FN_c[i] = cm[:, i].sum()-cm[i][i]

    OP = TP_c.sum() / (TP_c+FP_c).sum()
    PC = (TP_c/(TP_c+FP_c)).mean()
    mIoU = (TP_c/(TP_c+FP_c+FN_c)).mean()

    return OP, PC, mIoU


def cal_mIoU(cm):
    num_classes = cm.shape[0]

    TP_c = np.zeros(num_classes)
    for i in range(num_classes):
        TP_c[i] = cm[i][i]

    FP_c = np.zeros(num_classes)
    for i in range(num_classes):
        FP_c[i] = cm[i, :].sum()-cm[i][i]

    FN_c = np.zeros(num_classes)
    for i in range(num_classes):
        FN_c[i] = cm[:, i].sum()-cm[i][i]

    mIoU = (TP_c/(TP_c+FP_c+FN_c)).mean()

    return mIoU



class ProportionLoss(nn.Module):
    def __init__(self, metric="ce", eps=1e-8):
        super().__init__()
        self.metric = metric
        self.eps = eps

    def forward(self, input, target):
        if self.metric == "ce":
            loss = cross_entropy_loss(input, target, eps=self.eps) 

        elif self.metric == "ce_proposed":
            loss1 = cross_entropy_loss(input, target, eps=self.eps) 
            loss1 = torch.sum(loss1, dim=-1).mean()
            # loss2 = cross_entropy_loss(input.mean(dim=0), target.mean(dim=0), eps=self.eps)
            # loss2 = torch.sum(loss2, dim=-1).mean()
            loss = loss1
            if input.shape[0] // 2 == 2:
                num = input.shape[0] // 2
                loss3 = cross_entropy_loss(input[0:num].mean(dim=0), target[0:num].mean(dim=0), eps=self.eps)
                loss3 = torch.sum(loss3, dim=-1).mean()
                loss4 = cross_entropy_loss(input[num:].mean(dim=0), target[num:].mean(dim=0), eps=self.eps)
                loss4 = torch.sum(loss4, dim=-1).mean()
                loss34 = loss3 + loss4
                loss += loss34
            return loss

        elif self.metric == "l1":
            loss = F.l1_loss(input, target, reduction="none")
        elif self.metric == "mse":
            loss = F.mse_loss(input, target, reduction="none")
        else:
            raise NameError("metric {} is not supported".format(self.metric))

        loss = torch.sum(loss, dim=-1)
        loss = loss.mean()
        return loss


def load_data(args):  # Toy
    ######### load data #######
    test_data = np.load('data/%s/test_data.npy' % (args.dataset))
    test_label = np.load('data/%s/test_label.npy' % (args.dataset))
    bags = np.load('data/%s/bags.npy' % (args.dataset))
    labels = np.load('data/%s/labels.npy' % (args.dataset))
    lps = np.load('data/%s/origin_lps.npy' % (args.dataset))
    
    train_bags, val_bags, train_labels, val_labels, train_lps, val_lps = train_test_split(
        bags, labels, lps, test_size=args.split_ratio, random_state=42)
    
    unlabeled_bags = np.load('data/%s/unlabel_bags.npy' % (args.dataset))
    unlabeled_labels = np.load('data/%s/unlabel_labels.npy' % (args.dataset))
        # augment_bags = []
        # augment_labels = []
        # augment_lps = []
        # for i in range(train_bags.shape[0]):
        #     source, source_labels, source_lps = train_bags[i], train_labels[i], train_lps[i].reshape(1, args.classes)
        #     for j in range(train_bags.shape[0]):
        #         if i<=j:
        #             continue
        #         augment_bags.append(np.concatenate([source, train_bags[j]], axis=0))
        #         augment_labels.append(np.concatenate([source_labels, train_labels[j]], axis=0))
        #         augment_lps.append(np.concatenate([source_lps, train_lps[j].reshape(1, args.classes)], axis=0).mean(axis=0))
        # augment_bags = np.array(augment_bags)
        # augment_labels = np.array(augment_labels)
        # augment_lps = np.array(augment_lps)

        # train_bags = np.concatenate([train_bags, train_bags], axis=1)
        # train_labels = np.concatenate([train_labels, train_labels], axis=1)

        # train_bags = np.concatenate([train_bags, augment_bags], axis=0)
        # train_labels = np.concatenate([train_labels, augment_labels], axis=0)
        # train_lps = np.concatenate([train_lps, augment_lps], axis=0)

    train_dataset = DatasetBag(
        args=args, data=train_bags, label=train_labels, lp=train_lps, unlabel=unlabeled_bags, unlabel_label=unlabeled_labels, augment=args.augmentation)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.mini_batch,
        shuffle=True,  num_workers=args.num_workers)

    val_dataset = DatasetBag(
        args=args, data=val_bags, label=val_labels, lp=val_lps, unlabel=unlabeled_bags, unlabel_label=unlabeled_labels, augment=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.mini_batch,
        shuffle=False,  num_workers=args.num_workers)

    test_dataset = Dataset(data=test_data, label=test_label)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False,  num_workers=args.num_workers)

    return train_loader, val_loader, test_loader



def load_data_bags(args):  # Toy
    ######### load data #######
    test_data = np.load('data/%s/test_data.npy' % (args.dataset))
    test_label = np.load('data/%s/test_label.npy' % (args.dataset))
    # bags = np.load('data/%s/%d/bags.npy' % (args.dataset, args.fold))
    # labels = np.load('data/%s/%d/labels.npy' % (args.dataset, args.fold))
    # lps = np.load('data/%s/%d/origin_lps.npy' % (args.dataset, args.fold))

    train_bags = np.load('data/%s/%d/train_bags.npy' % (args.dataset, args.fold))
    train_labels = np.load('data/%s/%d/train_labels.npy' % (args.dataset, args.fold))
    train_lps = np.load('data/%s/%d/train_original_lps.npy' % (args.dataset, args.fold))  
    val_bags = np.load('data/%s/%d/val_bags.npy' % (args.dataset, args.fold)) 
    val_labels = np.load('data/%s/%d/val_labels.npy' % (args.dataset, args.fold))
    val_lps = np.load('data/%s/%d/val_original_lps.npy' % (args.dataset, args.fold))

    # train_bags, val_bags, train_labels, val_labels, train_lps, val_lps = train_test_split(
    # #     bags, labels, lps, test_size=args.split_ratio, random_state=42)
    
    # unlabeled_bags = np.load('data/%s/unlabel_bags.npy' % (args.dataset))
    # unlabeled_labels = np.load('data/%s/unlabel_labels.npy' % (args.dataset))
        # augment_bags = []
        # augment_labels = []
        # augment_lps = []
        # for i in range(train_bags.shape[0]):
        #     source, source_labels, source_lps = train_bags[i], train_labels[i], train_lps[i].reshape(1, args.classes)
        #     for j in range(train_bags.shape[0]):
        #         if i<=j:
        #             continue
        #         augment_bags.append(np.concatenate([source, train_bags[j]], axis=0))
        #         augment_labels.append(np.concatenate([source_labels, train_labels[j]], axis=0))
        #         augment_lps.append(np.concatenate([source_lps, train_lps[j].reshape(1, args.classes)], axis=0).mean(axis=0))
        # augment_bags = np.array(augment_bags)
        # augment_labels = np.array(augment_labels)
        # augment_lps = np.array(augment_lps)

        # train_bags = np.concatenate([train_bags, train_bags], axis=1)
        # train_labels = np.concatenate([train_labels, train_labels], axis=1)

        # train_bags = np.concatenate([train_bags, augment_bags], axis=0)
        # train_labels = np.concatenate([train_labels, augment_labels], axis=0)
        # train_lps = np.concatenate([train_lps, augment_lps], axis=0)

    train_dataset = DatasetBag_addbag(
        args=args, data=train_bags, label=train_labels, lp=train_lps, augment=args.augmentation)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.mini_batch,
        shuffle=True,  num_workers=args.num_workers)

    val_dataset = DatasetBag_addbag(
        args=args, data=val_bags, label=val_labels, lp=val_lps, augment=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.mini_batch,
        shuffle=False,  num_workers=args.num_workers)

    test_dataset = Dataset(data=test_data, label=test_label)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False,  num_workers=args.num_workers)

    return train_loader, val_loader, test_loader




class DatasetBag_randomchoice(torch.utils.data.Dataset):
    def __init__(self, args, data, label, lp, augment=True):
        np.random.seed(args.seed)
        self.data = data
        self.label = label
        self.lp = lp
        self.augment = augment

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        choice_bags = self.data[np.random.randint(0, self.len)]
        data = self.data[idx]
        label = self.label[idx]
        label = torch.tensor(label).long()

        index_order1 =np.arange(data.shape[0])
        random.shuffle(index_order1)
        index_order2 =np.arange(data.shape[0])
        random.shuffle(index_order2)

        data = data[index_order1]
        label = label[index_order1]
        choice_bags = choice_bags[index_order2]
        
        (b, w, h, c) = data.shape
        trans_data = torch.zeros((b, c, w, h))
        for i in range(b):
            trans_data[i] = self.transform(data[i])
        data = trans_data

        lp = self.lp[idx]
        lp = torch.tensor(lp).float()

        return data, label, lp


class DatasetBag_addbag(torch.utils.data.Dataset):
    def __init__(self, args, data, label, lp, augment=True):
        np.random.seed(args.seed)
        self.data = data
        self.label = label
        self.lp = lp
        self.augment = augment

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # choice_bags = self.data[np.random.randint(0, self.len)]
        data = self.data[idx]
        (b, w, h, c) = data.shape
        trans_data = torch.zeros((b, c, w, h))
        for i in range(b):
            trans_data[i] = self.transform(data[i])
        data = trans_data

        label = self.label[idx]
        label = torch.tensor(label).long()

        lp = self.lp[idx]
        lp = torch.tensor(lp).float()

        return data, label, lp