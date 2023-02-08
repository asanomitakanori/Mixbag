import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from losses import cross_entropy_loss
import torch.nn.functional as F
from scipy.stats import norm
from hydra.utils import to_absolute_path as to_abs_path 

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm

def worker_init_fn(seed):
    random.seed(seed)
    np.random.seed(seed)


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071), (0.2673))])

        self.len = self.data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        if len(data.shape) != 3:
            data = data.reshape(data.shape[0], data.shape[1], 1)
            data = self.transform2(data)
        else:
            data = self.transform(data)
        label = self.label[idx]
        label = torch.tensor(label).long()
        return data, label


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

    if args.statistic:
        if args.wrong_plp == False and args.correct_label == False:
            train_dataset = DatasetBag_randomchoice_statistic(
                                                            args=args, 
                                                            data=train_bags, 
                                                            label=train_labels, 
                                                            lp=train_lps, 
                                                            )
        elif args.wrong_plp == False and args.correct_label == True:
            train_dataset = DatasetBag_correctlabel(
                                                    args=args, 
                                                    data=train_bags, 
                                                    label=train_labels, 
                                                    lp=train_lps, 
                                                    )
        elif args.wrong_plp == True:
            train_dataset = DatasetBag_randomchoice_woplp(
                                                        args=args, 
                                                        data=train_bags, 
                                                        label=train_labels, 
                                                        lp=train_lps, 
                                                        augment=args.augmentation
                                                        )
    else:
        train_dataset = DatasetBag_train(
                                    args=args, 
                                    data=train_bags, 
                                    label=train_labels, 
                                    lp=train_lps,
                                    )
    train_loader = torch.utils.data.DataLoader(
                                    train_dataset, 
                                    batch_size=args.mini_batch, 
                                    worker_init_fn = worker_init_fn(args.seed),
                                    shuffle=True,  
                                    num_workers=args.num_workers
                                    )

    val_dataset = DatasetBag_val(
                            args=args, 
                            data=val_bags, 
                            label=val_labels, 
                            lp=val_lps, 
                            )
    val_loader = torch.utils.data.DataLoader(
                                    val_dataset, 
                                    batch_size=args.mini_batch,
                                    shuffle=False,  
                                    num_workers=args.num_workers
                                    )

    test_dataset = Dataset(data=test_data, label=test_label)
    test_loader = torch.utils.data.DataLoader(
                                    test_dataset, 
                                    batch_size=args.batch_size,
                                    shuffle=False,  
                                    num_workers=args.num_workers
                                    )

    return train_loader, val_loader, test_loader



class DatasetBag_train(torch.utils.data.Dataset):
    def __init__(self, args, data, label, lp):
        np.random.seed(args.seed)
        self.data = data
        self.label = label
        self.lp = lp
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071), (0.2673)),
            ])
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
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

        lp = self.lp[idx]
        lp = torch.tensor(lp).float()

        return data, label, lp


class DatasetBag_val(torch.utils.data.Dataset):
    def __init__(self, args, data, label, lp):
        np.random.seed(args.seed)
        self.data = data
        self.label = label
        self.lp = lp

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071), (0.2673))])
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
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

        lp = self.lp[idx]
        lp = torch.tensor(lp).float()

        return data, label, lp


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
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ])

        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071), (0.2673)),
            ])

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
            return data, label, lp, min_error, max_error

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
            return data, label, lp, min_error, max_error


class DatasetBag_randomchoice_woplp(torch.utils.data.Dataset):
    def __init__(self, args, data, label, lp, augment=True):
        fix_seed(args.seed)
        self.data = data
        self.label = label
        self.lp = lp
        self.augment = augment
        self.choice = args.choice

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071), (0.2673))])
        self.len = len(self.data)

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
                a: float = len(index_order1) / (len(index_order1) + len(index_order2))
                b: float =  len(index_order2) / (len(index_order1) + len(index_order2))
                lp: float = a*lp + b*choice_lp
                lp = torch.tensor(lp).float()
            elif self.choice == 'uniform':
                x = np.random.randint(1, data.shape[0])
                index_order1 = index_order1[0:x]
                index_order2 = index_order2[x:]
                a: float = len(index_order1) / (len(index_order1) + len(index_order2))
                b: float =  len(index_order2) / (len(index_order1) + len(index_order2))
                lp: float = a*lp + b*choice_lp
                lp = torch.tensor(lp).float()
            elif self.choice == 'gauss':
                x = np.random.normal(loc=0.5, scale=0.25, size=1)
                x = x * data.shape[0]
                index_order1 = index_order1[0:x]
                index_order2 = index_order2[x:]
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

            return data, label, lp
        
        else:
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
            # (b, w, h, c) = data.shape
            # trans_data = torch.zeros((b, c, w, h))
            # for i in range(b):
            #     trans_data[i] = self.transform(data[i])
            data = trans_data

            label = self.label[idx]
            label = torch.tensor(label).long()

            lp = self.lp[idx]
            lp = torch.tensor(lp).float()
            return data, label, lp
    

class DatasetBag_correctlabel(torch.utils.data.Dataset):
    def __init__(self, args, data, label, lp, augment=True):
        fix_seed(args.seed)
        self.data = data
        self.label = label
        self.lp = lp
        self.augment = augment
        self.choice = args.choice
        self.classes = args.classes

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071), (0.2673))])
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        n = np.random.rand()
        if n >= 0.5:
            choice_bags_num = np.random.randint(0, self.len)
            choice_bags, choice_labels = self.data[choice_bags_num], self.label[choice_bags_num]
            data = self.data[idx]
            label = self.label[idx]
            index_order1 =np.arange(data.shape[0])
            random.shuffle(index_order1)
            index_order2 =np.arange(data.shape[0])
            random.shuffle(index_order2)

            if self.choice == 'half':
                index_order1 = index_order1[0:index_order1.shape[0] // 2]
                index_order2 = index_order2[0:index_order2.shape[0] // 2]
            elif self.choice == 'uniform':
                x = np.random.randint(1, data.shape[0])
                index_order1 = index_order1[0:x]
                index_order2 = index_order2[x:]
            elif self.choice == 'gauss':
                x = np.random.normal(loc=0.5, scale=0.25, size=1)
                x = int(x * data.shape[0])
                if x > data.shape[0]:
                    x = data.shape[0] - 1
                elif x < 0:
                    x = 1
                index_order1 = index_order1[0:x]
                index_order2 = index_order2[x:]

            data = data[index_order1]
            label = label[index_order1]
            choice_bags = choice_bags[index_order2]
            choice_labels = choice_labels[index_order2]
            data = np.concatenate([data, choice_bags], axis=0)
            label = np.concatenate([label, choice_labels])
            label = torch.tensor(label).long()

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
            lp = np.array(lp_list)
            lp = lp / label.shape[0]
            lp = torch.tensor(lp).float()
            return data, label, lp
        
        else:
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
            # (b, w, h, c) = data.shape
            # trans_data = torch.zeros((b, c, w, h))
            # for i in range(b):
            #     trans_data[i] = self.transform(data[i])
            data = trans_data

            label = self.label[idx]
            label = torch.tensor(label).long()

            lp = self.lp[idx]
            lp = torch.tensor(lp).float()
            return data, label, lp