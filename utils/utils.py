import os
import random
import seaborn as sns

import numpy as np
from scipy.stats import norm
from hydra.utils import to_absolute_path as to_abs_path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.models import resnet18


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm


def worker_init_fn(seed):
    random.seed(seed)
    np.random.seed(seed)


def save_confusion_matrix(cm, path, title=""):
    sns.heatmap(cm, annot=True, cmap="Blues_r", fmt=".4f")
    plt.xlabel("pred")
    plt.ylabel("GT")
    plt.title(title)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def load_data_bags(args):  # Toy
    ######### load data #######
    test_data = np.load(to_abs_path("data/%s/test_data.npy" % (args.dataset)))
    test_label = np.load(to_abs_path("data/%s/test_label.npy" % (args.dataset)))

    train_bags = np.load(
        to_abs_path("data/%s/%d/train_bags.npy" % (args.dataset, args.fold))
    )
    train_labels = np.load(
        to_abs_path("data/%s/%d/train_labels.npy" % (args.dataset, args.fold))
    )
    train_lps = np.load(
        to_abs_path("data/%s/%d/train_original_lps.npy" % (args.dataset, args.fold))
    )
    val_bags = np.load(
        to_abs_path("data/%s/%d/val_bags.npy" % (args.dataset, args.fold))
    )
    val_labels = np.load(
        to_abs_path("data/%s/%d/val_labels.npy" % (args.dataset, args.fold))
    )
    val_lps = np.load(
        to_abs_path("data/%s/%d/val_original_lps.npy" % (args.dataset, args.fold))
    )

    train_dataset = Dataset_Mixbag(
        args=args,
        data=train_bags,
        label=train_labels,
        lp=train_lps,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.mini_batch,
        worker_init_fn=worker_init_fn(args.seed),
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_dataset = Dataset_Val(
        args=args,
        data=val_bags,
        label=val_labels,
        lp=val_lps,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.mini_batch,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_dataset = Dataset_Test(data=test_data, label=test_label)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader, test_loader


class Dataset_Val(torch.utils.data.Dataset):
    def __init__(self, args, data, label, lp):
        np.random.seed(args.seed)
        self.data = data
        self.label = label
        self.lp = lp

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
                ),
            ]
        )
        self.transform2 = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5071), (0.2673))]
        )
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

        return {"img": data, "label": label, "label_prop": lp}


class Dataset_Test(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
                ),
            ]
        )
        self.transform2 = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5071), (0.2673))]
        )
        self.len = self.data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img = self.data[idx]
        if len(img.shape) != 3:
            img = img.reshape(img.shape[0], img.shape[1], 1)
            img = self.transform2(img)
        else:
            img = self.transform(img)
        label = self.label[idx]
        label = torch.tensor(label).long()
        return {"img": img, "label": label}


class Dataset_Mixbag(torch.utils.data.Dataset):
    def __init__(self, args, data, label, lp):
        fix_seed(args.seed)
        self.CI = args.confidence_interval
        self.data = data
        self.label = label
        self.lp = lp
        self.classes = args.classes

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
                ),
            ]
        )
        self.transform2 = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5071), (0.2673))]
        )
        self.len = len(self.data)
        self.choice = args.choice

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        n = np.random.rand()
        if n >= 0.5:
            choice_bags_index = np.random.randint(0, self.len)
            choice_bags, choice_labels, choice_lp = (
                self.data[choice_bags_index],
                self.label[choice_bags_index],
                self.lp[choice_bags_index],
            )
            data, label, lp = self.data[idx], self.label[idx], self.lp[idx]
            index_order1, index_order2 = np.arange(data.shape[0]), np.arange(
                data.shape[0]
            )
            random.shuffle(index_order1)
            random.shuffle(index_order2)

            if self.choice == "half":
                index_order1 = index_order1[0 : index_order1.shape[0] // 2]
                index_order2 = index_order2[0 : index_order2.shape[0] // 2]
                min_error, max_error, lp = error_cover_area(
                    lp, choice_lp, len(index_order1), len(index_order2), self.CI
                )
                lp = torch.tensor(lp).float()

            elif self.choice == "uniform":
                x = np.random.randint(1, data.shape[0])
                index_order1 = index_order1[0:x]
                index_order2 = index_order2[x:]
                min_error, max_error, lp = error_cover_area(
                    lp, choice_lp, len(index_order1), len(index_order2), self.CI
                )
                lp = torch.tensor(lp).float()

            elif self.choice == "gauss":
                x = np.random.normal(loc=0.5, scale=0.1, size=1)
                x = int(x * data.shape[0])
                if x == 0:
                    x = 1
                elif x >= data.shape[0]:
                    x = data.shape[-1]
                index_order1 = index_order1[0:x]
                index_order2 = index_order2[x:]
                min_error, max_error, lp = error_cover_area(
                    lp, choice_lp, len(index_order1), len(index_order2), self.CI
                )
                lp = torch.tensor(lp).float()

            data, label = data[index_order1], label[index_order1]
            choice_bags, choice_labels = (
                choice_bags[index_order2],
                choice_labels[index_order2],
            )
            data = np.concatenate([data, choice_bags], axis=0)
            label = np.concatenate([label, choice_labels])
            label = torch.tensor(label).long()
            min_error, max_error = (
                torch.tensor(min_error).float(),
                torch.tensor(max_error).float(),
            )

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

            return {
                "img": data,
                "label": label,
                "label_prop": lp,
                "ci_min_value": min_error,
                "ci_max_value": max_error,
            }

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
            min_error, max_error = (
                torch.full((1, self.classes), -1).reshape(self.classes).float(),
                torch.full((1, self.classes), -1).reshape(self.classes).float(),
            )
            lp = self.lp[idx]
            lp = torch.tensor(lp).float()

            return {
                "img": data,
                "label": label,
                "label_prop": lp,
                "ci_min_value": min_error,
                "ci_max_value": max_error,
            }


def error_cover_area(
    proportion1: list,
    proportion2: list,
    sampling_num1: int,
    sampling_num2: int,
    confidence_interval: float,
):
    a: float = sampling_num1 / (sampling_num1 + sampling_num2)
    b: float = sampling_num2 / (sampling_num1 + sampling_num2)
    t = norm.isf(q=confidence_interval)
    cover1 = t * np.sqrt(proportion1 * (1 - proportion1) / sampling_num1)
    cover2 = t * np.sqrt(proportion2 * (1 - proportion2) / sampling_num2)
    expected_plp = a * proportion1 + b * proportion2
    confidence_area = t * cover1 + b * cover2
    min = expected_plp - confidence_area
    max = expected_plp + confidence_area
    return min, max, expected_plp


def model_import(args, model_name=None):
    model = resnet18(pretrained=args.pretrained)
    if model:
        if args.channels != 3:
            model.conv1 = nn.Conv2d(
                args.channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        model.fc = nn.Linear(model.fc.in_features, args.classes)
        model = model.to(args.device)
    return model


if __name__ == "__main__":
    proportion = [0.2, 0.7, 0.1]
    sampling_num = 128
    confidence_interval = 0.025
    error = error_cover_area(proportion, sampling_num, confidence_interval)
    print(error)
